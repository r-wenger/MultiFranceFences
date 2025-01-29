import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch import Unet
from torch.utils.tensorboard import SummaryWriter
#from data_loader_concatenate import get_data_loaders
from data_loader import get_data_loaders
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from loss import dice_coefficient

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=25, lr_decay_factor=0.75, decay_epochs=5, patience=10, output_mode="both"):
    """
    Train a model with early stopping and manual learning rate adjustment, supporting different DataLoader output modes.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training.
        valid_loader (DataLoader): DataLoader for validation.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to use (CPU or GPU).
        num_epochs (int): Total number of epochs.
        lr_decay_factor (float): Multiplicative factor to adjust the learning rate.
        decay_epochs (int): Number of epochs between learning rate adjustments.
        patience (int): Number of epochs to wait for improvement in validation loss before stopping.
        output_mode (str): Mode of DataLoader output. Options: "both", "image", "dsm", "concat".

    Returns:
        torch.nn.Module: The trained model.
    """
    model.train()
    writer = SummaryWriter()

    train_losses = []
    valid_losses = []
    train_dices = []
    valid_dices = []

    best_loss = float('inf')  # Best validation loss
    best_model = None         # Best model state
    early_stop_counter = 0    # Counter for early stopping

    for epoch in range(num_epochs):
        # Adjust learning rate manually every `decay_epochs` epochs
        if epoch > 0 and epoch % decay_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor
            print(f"[INFO] Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")

        # Training
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        for batch in train_loader:
            if output_mode == "both":
                (ortho_images, lidar_images), masks, _ = batch
                input_ortho = ortho_images.to(device)
                input_dsm = lidar_images.to(device)
            elif output_mode == "image":
                ortho_images, masks, _ = batch
                inputs = ortho_images.to(device)
            elif output_mode == "dsm":
                lidar_images, masks, _ = batch
                inputs = lidar_images.to(device)
            elif output_mode == "concat":
                concat_images, masks, _ = batch
                inputs = concat_images.to(device)
            else:
                raise ValueError(f"Unknown output_mode: {output_mode}")

            masks = masks.to(device)

            optimizer.zero_grad()
            if output_mode == 'both':
                outputs = model(input_ortho, input_dsm)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * masks.size(0)
            running_dice += dice_coefficient(outputs, masks) * masks.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_dice = running_dice / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_dices.append(epoch_dice)

        print(f"Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}")
        writer.add_scalar('training_loss', epoch_loss, epoch)
        writer.add_scalar('training_dice', epoch_dice, epoch)

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_dice = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                if output_mode == "both":
                    (ortho_images, lidar_images), masks, _ = batch
                    input_ortho = ortho_images.to(device)
                    input_dsm = lidar_images.to(device)
                elif output_mode == "image":
                    ortho_images, masks, _ = batch
                    inputs = ortho_images.to(device)
                elif output_mode == "dsm":
                    lidar_images, masks, _ = batch
                    inputs = lidar_images.to(device)
                elif output_mode == "concat":
                    concat_images, masks, _ = batch
                    inputs = concat_images.to(device)
                else:
                    raise ValueError(f"Unknown output_mode: {output_mode}")

                masks = masks.to(device)

                if output_mode == 'both':
                    outputs = model(input_ortho, input_dsm)
                else:
                    outputs = model(inputs)
                    
                loss = criterion(outputs, masks)

                valid_loss += loss.item() * masks.size(0)
                valid_dice += dice_coefficient(outputs, masks) * masks.size(0)

        valid_loss /= len(valid_loader.dataset)
        valid_dice /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        valid_dices.append(valid_dice)

        print(f"Validation Loss: {valid_loss:.4f}, Validation Dice: {valid_dice:.4f}")
        writer.add_scalar('validation_loss', valid_loss, epoch)
        writer.add_scalar('validation_dice', valid_dice, epoch)

        # Early Stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model.state_dict()  # Save the best model state
            early_stop_counter = 0
            print("[INFO] Validation loss improved. Model saved.")
        else:
            early_stop_counter += 1
            print(f"[INFO] No improvement in validation loss for {early_stop_counter} epoch(s).")

        if early_stop_counter >= patience:
            print("[INFO] Early stopping triggered.")
            break

    writer.close()

    # Load the best model before returning
    if best_model:
        model.load_state_dict(best_model)

    return model
