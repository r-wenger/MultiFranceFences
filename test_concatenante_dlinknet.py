import os
import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch import Unet
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from dinknet import DinkNet101
from sklearn.metrics import classification_report, confusion_matrix
import rasterio
from data_loader_ram import get_data_loaders
from tqdm import tqdm

ortho_means = [31.63289965, 34.25222465, 31.6679167]  # Replace with actual calculated means for each band
ortho_stds = [45.82212139, 46.20130014, 42.59491153]  # Replace with actual calculated stds for each band
lidar_mean = 483.59665233887523  # Replace with actual calculated mean
lidar_std = 480.7912533243046   # Replace with actual calculated std

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    all_predictions = []
    all_masks = []
    all_image_names = []

    with torch.no_grad():
        for images, masks, image_names in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (outputs.sigmoid() > threshold).float()
            
            all_predictions.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            all_image_names.extend(image_names)  # Collect all image names

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    return all_predictions, all_masks, all_image_names


def calculate_metrics(predictions, masks):
    preds_flat = predictions.flatten()
    masks_flat = masks.flatten()

    report = classification_report(masks_flat, preds_flat, labels=[0, 1], target_names=['background', 'fence'], output_dict=True)
    tn, fp, fn, tp = confusion_matrix(masks_flat, preds_flat, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0

    return precision, recall, iou, dice, report

def normalize_and_clip(image, mean, std):
    image = (image * std) + mean
    image = np.clip(image, 0, 255)  
    return image

def save_visualizations(predictions, masks, ortho_images, lidar_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, (pred, mask, ortho_image, lidar_image) in enumerate(zip(predictions, masks, ortho_images, lidar_images)):
        ortho_image = ortho_image[:, :, :3].astype(np.float32)
        for j in range(3):
            ortho_image[:, :, j] = normalize_and_clip(ortho_image[:, :, j], ortho_means[j], ortho_stds[j])

        lidar_image = (lidar_image * lidar_std) + lidar_mean

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        axs[0].imshow(ortho_image.astype(np.uint8))
        axs[0].set_title("RGB Image")
        axs[0].axis('off')

        axs[1].imshow(lidar_image.squeeze(), cmap='gray')
        axs[1].set_title("Lidar Image")
        axs[1].axis('off')

        axs[2].imshow(mask.squeeze(), cmap='gray')
        axs[2].set_title("Ground Truth")
        axs[2].axis('off')

        axs[3].imshow(pred.squeeze(), cmap='gray')
        axs[3].set_title("Prediction")
        axs[3].axis('off')

        plt.savefig(os.path.join(output_dir, f"result_{i}.png"))
        plt.close()


def save_results_as_tif(predictions, masks, img_names, ortho_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for i, (pred, mask, img_name) in enumerate(zip(predictions, masks, img_names)):
        # Open the original ortho image to get the transform and CRS
        with rasterio.open(os.path.join(ortho_path, img_name)) as src:
            transform = src.transform
            crs = src.crs
            height, width = pred.shape[1], pred.shape[2]
            channels_pred = pred.shape[0]  # Number of channels in the prediction
            channels_mask = mask.shape[0]  # Number of channels in the mask

        base_name = os.path.splitext(img_name)[0]
        # Construct the file paths using the base name
        pred_tif_path = os.path.join(output_dir, f"{base_name}_prediction.tif")
        mask_tif_path = os.path.join(output_dir, f"{base_name}_mask.tif")

        with rasterio.open(
            pred_tif_path, 
            'w', 
            driver='GTiff', 
            height=height, 
            width=width,
            count=channels_pred,  # Number of channels in the prediction
            dtype='float32', 
            crs=crs, 
            transform=transform
        ) as dst:
            dst.write(pred, list(range(1, channels_pred + 1)))

        with rasterio.open(
            mask_tif_path, 
            'w', 
            driver='GTiff', 
            height=height, 
            width=width,
            count=channels_mask,  # Number of channels in the mask
            dtype='float32', 
            crs=crs, 
            transform=transform
        ) as dst:
            dst.write(mask, list(range(1, channels_mask + 1)))


def find_best_and_worst_patches(predictions, masks):
    f1_scores = []

    for pred, mask in zip(predictions, masks):
        pred_flat = pred.flatten()
        mask_flat = mask.flatten()

        tp = np.sum((pred_flat == 1) & (mask_flat == 1))
        fp = np.sum((pred_flat == 1) & (mask_flat == 0))
        fn = np.sum((pred_flat == 0) & (mask_flat == 1))

        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    mean_f1 = np.mean(f1_scores)
    
    best_patch_idx = np.argmax(f1_scores)
    worst_patch_idx = np.argmin(f1_scores)
    closest_to_mean_idx = np.argmin(np.abs(f1_scores - mean_f1))

    return best_patch_idx, closest_to_mean_idx, worst_patch_idx, f1_scores

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the test data loader
    ortho_dir = "./dataset_fences_france_filtered/ortho"
    lidar_dir = "./dataset_fences_france_filtered/lidar"
    mask_dir = "./dataset_fences_france_filtered/fences_2m"
    gpkg_path = './Departments/departements_grouped.gpkg'
    batch_size = 16
    num_workers = 4
    output_dir = "" #Dir with your results corresponding to this test
    use_lidar = False
    in_channels = 3
    
    # Load the trained model
    print('[INFO] Loading weights ...')
    model = DinkNet101(num_classes=1, n_bands=in_channels)
    model.load_state_dict(torch.load('./your_test.pth'))
    model = model.to(device)
    print('[INFO] Weights loaded.')

    os.makedirs(output_dir, exist_ok=True)

    print('[INFO] Data loader ...')
    _, _, test_loader = get_data_loaders(ortho_dir, lidar_dir, mask_dir, gpkg_path, batch_size, num_workers, use_lidar)

    # Evaluate the model
    print('[INFO] Evaluation ...')
    predictions, masks, image_names = evaluate_model(model, test_loader, device)
    precision, recall, iou, dice, report = calculate_metrics(predictions, masks)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")
    print("Classification Report:")
    print(report)

    image_names = test_loader.dataset.image_names

    best_patch_idx, closest_to_mean_idx, worst_patch_idx, f1_scores = find_best_and_worst_patches(predictions, masks)

    best_patch_info = {
        "f1_score": f1_scores[best_patch_idx],
        "patch_index": best_patch_idx,
        "ortho_path": test_loader.dataset.image_names[best_patch_idx],
        "lidar_path": test_loader.dataset.image_names[best_patch_idx]
    }

    mean_patch_info = {
        "f1_score": f1_scores[closest_to_mean_idx],
        "patch_index": closest_to_mean_idx,
        "ortho_path": test_loader.dataset.image_names[closest_to_mean_idx],
        "lidar_path": test_loader.dataset.image_names[closest_to_mean_idx]
    }

    worst_patch_info = {
        "f1_score": f1_scores[worst_patch_idx],
        "patch_index": worst_patch_idx,
        "ortho_path": test_loader.dataset.image_names[worst_patch_idx],
        "lidar_path": test_loader.dataset.image_names[worst_patch_idx]
    }

    os.makedirs(os.path.join(output_dir, 'save_viz'), exist_ok=True)
    ortho_images = [x[0] for x in test_loader.dataset]
    lidar_images = [x[1] for x in test_loader.dataset]
    save_visualizations(predictions, masks, ortho_images, lidar_images, output_dir=os.path.join(output_dir, 'save_viz'))

    os.makedirs(os.path.join(output_dir, 'save_tif'), exist_ok=True)

    save_results_as_tif(
        predictions, 
        masks,
        image_names,
        ortho_path=ortho_dir, 
        output_dir=os.path.join(output_dir, 'save_tif')
    )

    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(str(report))
        f.write("\n\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"IoU: {iou:.4f}\n")
        f.write(f"Dice Coefficient: {dice:.4f}\n")
        f.write("\n\n")

        f.write("Best Patch Information:\n")
        f.write(f"F1 Score: {best_patch_info['f1_score']:.4f}\n")
        f.write(f"Patch Index: {best_patch_info['patch_index']}\n")
        f.write(f"Ortho Path: {best_patch_info['ortho_path']}\n")
        f.write(f"Lidar Path: {best_patch_info['lidar_path']}\n")
        f.write("\n\n")

        f.write("Patch Closest to Mean F1 Score:\n")
        f.write(f"F1 Score: {mean_patch_info['f1_score']:.4f}\n")
        f.write(f"Patch Index: {mean_patch_info['patch_index']}\n")
        f.write(f"Ortho Path: {mean_patch_info['ortho_path']}\n")
        f.write(f"Lidar Path: {mean_patch_info['lidar_path']}\n")
        f.write("\n\n")

        f.write("Worst Patch Information:\n")
        f.write(f"F1 Score: {worst_patch_info['f1_score']:.4f}\n")
        f.write(f"Patch Index: {worst_patch_info['patch_index']}\n")
        f.write(f"Ortho Path: {worst_patch_info['ortho_path']}\n")
        f.write(f"Lidar Path: {worst_patch_info['lidar_path']}\n")

if __name__ == "__main__":
    main()
