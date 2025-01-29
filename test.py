import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import rasterio


def evaluate_model(model, test_loader, device, threshold=0.5, output_mode="both"):
    model.eval()
    all_predictions = []
    all_masks = []
    all_image_names = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if output_mode == "both":
                (ortho_images, lidar_images), masks, image_names = batch
                input_ortho = ortho_images.to(device)
                input_dsm = lidar_images.to(device)
            elif output_mode == "image":
                ortho_images, masks, image_names = batch
                inputs = ortho_images.to(device)
            elif output_mode == "dsm":
                lidar_images, masks, image_names = batch
                inputs = lidar_images.to(device)
            elif output_mode == "concat":
                concat_images, masks, image_names = batch
                inputs = concat_images.to(device)
            else:
                raise ValueError(f"Unknown output_mode: {output_mode}")

            masks = masks.to(device)
            
            if output_mode == "both":
                outputs = model(input_ortho, input_dsm)
            else:
                outputs = model(inputs)
            preds = (outputs.sigmoid() > threshold).float()

            all_predictions.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
            all_image_names.extend(image_names)

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    return all_predictions, all_masks, all_image_names


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
        pred_tif_path = os.path.join(output_dir, f"{base_name}_prediction.tif")
        mask_tif_path = os.path.join(output_dir, f"{base_name}_mask.tif")

        # Save the prediction
        with rasterio.open(
            pred_tif_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=channels_pred,
            dtype='float32',
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(pred, list(range(1, channels_pred + 1)))

        # Save the mask
        with rasterio.open(
            mask_tif_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=channels_mask,
            dtype='float32',
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(mask, list(range(1, channels_mask + 1)))


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


def save_results(predictions, masks, image_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, 'classification_report.txt')
    precision, recall, iou, dice, report = calculate_metrics(predictions, masks)

    # Sauvegarder les métriques
    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(str(report))
        f.write("\n\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"IoU: {iou:.4f}\n")
        f.write(f"Dice Coefficient: {dice:.4f}\n")

    print(f"[INFO] Results saved in {output_dir}")
