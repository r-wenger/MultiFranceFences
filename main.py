import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch import Unet
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_data_loaders
import numpy as np
import matplotlib.pyplot as plt
from model import UNet, UNetLateFusion
from datetime import datetime
from loss import get_loss_function
from train import train_model
import os

class Config:
    def __init__(self):
        # Général
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ortho_dir = "/a2s/rwenger/work/polymorFENCE/dataset_fences_france_filtered/ortho"
        self.lidar_dir = "/a2s/rwenger/work/polymorFENCE/dataset_fences_france_filtered/lidar"
        self.mask_dir = "/a2s/rwenger/work/polymorFENCE/dataset_fences_france_filtered/fences_3m"
        self.gpkg_path = '/a2s/rwenger/work/polymorFENCE/Departments/departements_grouped.gpkg'
        self.models_path = '/a2s/rwenger/work/polymorFENCE/deep_learning_clean/models'
        self.outname_base = 'unet_france_3m'
        self.use_lidar = True

        # Modèle
        self.network = 'UNetConcatenate' #or 'UNetLate' or 'UNetConcatenate' or 'UNetDSM' or 'UNetRGB'
        self.decoder_channels = [512, 256, 128, 64]
        self.n_classes = 1
        self.lr_decay_factor = 0.75  # Reduce LR by 25%
        self.decay_epochs = 5        # Every 5 epoch

        # Entraînement
        self.sampling = 'random' #'random' or 'geographic' 
        self.batch_size = 16
        self.num_workers = 4
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.patience = 10

        # Perte
        self.loss_type = 'dice'  # 'dice' or 'focal' or 'combined'
        
        # Optimiseur et scheduler
        self.optimizer = 'Adam'

    def generate_outname(self):
        """Génère un nom de fichier .pth basé sur les paramètres."""
        lidar_suffix = "_lidar" if self.use_lidar else "_no_lidar"
        loss_suffix = f"_{self.loss_type}_loss"
        date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.outname_base}_{self.sampling}_{self.network}_bs{self.batch_size}_lr{self.learning_rate}{lidar_suffix}{loss_suffix}_{date_suffix}.pth"


def main(config):
    print(f"Using device: {config.device}")

    print("[INFO] Training configuration:")
    print(config.__dict__)

    print("[INFO] Creating model ...")
    if config.network == 'UNetRGB':
        model = UNet(input_channels=3, decoder_channels=config.decoder_channels, n_classes=1).to(config.device)
        output_mode_data_loader = 'image'
    elif config.network == 'UNetLate':
        model = UNetLateFusion(decoder_channels=config.decoder_channels, n_classes=1).to(config.device)
        output_mode_data_loader = 'both'
    elif config.network == 'UNetConcatenate':
        model = UNet(input_channels=4, decoder_channels=config.decoder_channels, n_classes=1).to(config.device)
        output_mode_data_loader = 'concat'
    elif config.network == 'UNetDSM':
        model = UNet(input_channels=1, decoder_channels=config.decoder_channels, n_classes=1).to(config.device)
        output_mode_data_loader = 'dsm'

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    print("[INFO] End creating model.")

    print("[INFO] Loading data ...")
    train_loader, valid_loader, test_loader = get_data_loaders(
        config.ortho_dir, config.lidar_dir, config.mask_dir, config.gpkg_path, config.batch_size, num_workers=4, 
        stratification_method=config.sampling, random_seed=42, output_mode=output_mode_data_loader
    )
    print("[INFO] End loading data.")

    criterion = get_loss_function(config.loss_type)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"[INFO] Using {config.loss_type} loss.")
    print("[INFO] Training ...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        num_epochs=config.num_epochs,
        lr_decay_factor=config.lr_decay_factor,
        decay_epochs=config.decay_epochs,
        patience=config.patience,  
        output_mode=output_mode_data_loader  
    )
    print("[INFO] End training.")

    outname = config.generate_outname()
    torch.save(trained_model.state_dict(), os.path.join(config.models_path ,outname))
    print(f"[INFO] Model saved as {outname}")


if __name__ == "__main__":
    config = Config()

    main(config)