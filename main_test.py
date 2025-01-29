import os
import torch
from test import evaluate_model, save_results_as_tif, find_best_and_worst_patches, evaluate_model, calculate_metrics
from data_loader import get_data_loaders
from model import UNet, UNetLateFusion
import re
from datetime import datetime
import torch
import os
from torch.nn import DataParallel


class Config:
    def __init__(self, model_name=None):
        # Général
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ortho_dir = "/a2s/rwenger/work/polymorFENCE/dataset_fences_france_filtered/ortho"
        self.lidar_dir = "/a2s/rwenger/work/polymorFENCE/dataset_fences_france_filtered/lidar"
        self.mask_dir = "/a2s/rwenger/work/polymorFENCE/dataset_fences_france_filtered/fences_3m"
        self.gpkg_path = '/a2s/rwenger/work/polymorFENCE/Departments/departements_grouped.gpkg'
        self.models_path = '/a2s/rwenger/work/polymorFENCE/deep_learning_clean/models'
        self.output_dir = os.path.join('/a2s/rwenger/work/polymorFENCE/deep_learning_clean/results/', os.path.splitext(model_name)[0], 'evaluation')
        self.outname_base = 'unet_france_3m'

        # Initialiser les paramètres par défaut
        self.network = None
        self.decoder_channels = [512, 256, 128, 64]
        self.n_classes = 1
        self.num_workers = 4
        self.sampling = None
        self.batch_size = None
        self.learning_rate = None
        self.loss_type = None
        self.lidar_suffix = None

        # Si un nom de fichier modèle est donné, parser les paramètres
        if model_name:
            self.parse_model_name(model_name)
            self.model_name = model_name
        else:
            self.model_name = None


    def parse_model_name(self, model_name):
        """
        Parse les paramètres depuis un nom de fichier modèle.
        Exemple : "unet_france_3m_random_UNetLate_bs16_lr0.0001_lidar_dice_loss_20250121_123456.pth"
        """
        pattern = (
            r"(?P<base>.+)_"               # Nom de base
            r"(?P<sampling>random|geographic)_"  # Méthode de sampling
            r"(?P<network>UNetLate|UNetConcatenate|UNetDSM|UNetRGB)_"  # Modèle
            r"bs(?P<batch_size>\d+)_"      # Taille des batches
            r"lr(?P<learning_rate>[\d\.]+)_"  # Learning rate
            r"(?P<lidar_suffix>lidar|no_lidar)_"  # Utilisation du lidar
            r"(?P<loss_type>\w+)_loss"     # Type de perte
        )
        match = re.match(pattern, model_name)
        if match:
            self.outname_base = match.group("base")
            self.sampling = match.group("sampling")
            self.network = match.group("network")
            self.batch_size = int(match.group("batch_size"))
            self.learning_rate = float(match.group("learning_rate"))
            self.lidar_suffix = match.group("lidar_suffix")
            self.loss_type = match.group("loss_type")
        else:
            raise ValueError(f"Invalid model name format: {model_name}")


    def generate_outname(self):
        """Génère un nom de fichier pour sauvegarder les résultats des tests."""
        date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.outname_base}_{self.sampling}_{self.network}_{date_suffix}.txt"


def main():
    model_name = "unet_france_3m_random_UNetLate_bs16_lr0.0001_lidar_dice_loss_20250123_221551.pth"
    config = Config(model_name=model_name)

    print("[INFO] Parsed Configuration:")
    print(config.__dict__)

    print("[INFO] Loading model...")
    if config.network == 'UNetLate':
        model = UNetLateFusion(decoder_channels=config.decoder_channels, n_classes=config.n_classes).to(config.device)
        output_mode_data_loader = 'both'
    elif config.network == 'UNetRGB':
        model = UNet(input_channels=3, decoder_channels=config.decoder_channels, n_classes=config.n_classes).to(config.device)
        output_mode_data_loader = 'image'
    elif config.network == 'UNetDSM':
        model = UNet(input_channels=1, decoder_channels=config.decoder_channels, n_classes=config.n_classes).to(config.device)
        output_mode_data_loader = 'dsm'
    elif config.network == 'UNetConcatenate':
        model = UNet(input_channels=4, decoder_channels=config.decoder_channels, n_classes=config.n_classes).to(config.device)
        output_mode_data_loader = 'concat'

    model_path = os.path.join(config.models_path, config.model_name)
    model = DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=config.device))

    print(f"[INFO] Model {config.model_name} loaded.")

    print("[INFO] Loading test data...")
    _, _, test_loader = get_data_loaders(
        config.ortho_dir, config.lidar_dir, config.mask_dir, config.gpkg_path,
        batch_size=config.batch_size, num_workers=config.num_workers,
        stratification_method=config.sampling, random_seed=42, output_mode=output_mode_data_loader
    )
    print("[INFO] Test data loaded.")

    print("[INFO] Evaluating model...")
    predictions, masks, image_names = evaluate_model(
        model=model, test_loader=test_loader, device=config.device, output_mode=output_mode_data_loader
    )

    precision, recall, iou, dice, report = calculate_metrics(predictions, masks)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")
    print(report)

    print("[INFO] Saving predictions and masks...")
    save_results_as_tif(predictions, masks, image_names, config.ortho_dir, config.output_dir)

    print("[INFO] Finding best and worst patches...")
    best_idx, mean_idx, worst_idx, f1_scores = find_best_and_worst_patches(predictions, masks)

    print(f"Best Patch Index: {best_idx}, F1 Score: {f1_scores[best_idx]:.4f}")
    print(f"Patch Closest to Mean Index: {mean_idx}, F1 Score: {f1_scores[mean_idx]:.4f}")
    print(f"Worst Patch Index: {worst_idx}, F1 Score: {f1_scores[worst_idx]:.4f}")

    report_path = os.path.join(config.output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(str(report))
        f.write("\n\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"IoU: {iou:.4f}\n")
        f.write(f"Dice Coefficient: {dice:.4f}\n")
        f.write("\n\n")
        f.write(f"Best Patch: {best_idx}, F1 Score: {f1_scores[best_idx]:.4f}\n")
        f.write(f"Mean Patch: {mean_idx}, F1 Score: {f1_scores[mean_idx]:.4f}\n")
        f.write(f"Worst Patch: {worst_idx}, F1 Score: {f1_scores[worst_idx]:.4f}\n")

    print("[INFO] Evaluation complete. Results saved.")


if __name__ == "__main__":
    main()
