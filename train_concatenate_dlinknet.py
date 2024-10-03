import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch import Unet
from torch.utils.tensorboard import SummaryWriter
from data_loader_ram import get_data_loaders
import numpy as np
import matplotlib.pyplot as plt
from dinknet import DinkNet101

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def dice_coefficient(outputs, masks, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()
    smooth = 1e-5

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def iou(outputs, masks, threshold=0.5):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > threshold).float()
    smooth = 1e-5

    intersection = (preds * masks).sum()
    total = preds.sum() + masks.sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=25, outname="test"):
    model.train()
    writer = SummaryWriter()

    train_losses = []
    valid_losses = []
    train_dices = []
    valid_dices = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_dice += dice_coefficient(outputs, masks) * images.size(0)
            running_iou += iou(outputs, masks) * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_dice = running_dice / len(train_loader.dataset)
        epoch_iou = running_iou / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_dices.append(epoch_dice)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}')
        writer.add_scalar('training_loss', epoch_loss, epoch)
        writer.add_scalar('training_dice', epoch_dice, epoch)
        writer.add_scalar('training_iou', epoch_iou, epoch)

        # Validation step
        model.eval()
        valid_loss = 0.0
        valid_dice = 0.0
        valid_iou = 0.0
        with torch.no_grad():
            for images, masks, _ in valid_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                valid_loss += loss.item() * images.size(0)
                valid_dice += dice_coefficient(outputs, masks) * images.size(0)
                valid_iou += iou(outputs, masks) * images.size(0)

        valid_loss /= len(valid_loader.dataset)
        valid_dice /= len(valid_loader.dataset)
        valid_iou /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        valid_dices.append(valid_dice)
        print(f'Validation Loss: {valid_loss:.4f}, Validation Dice: {valid_dice:.4f}, Validation IoU: {valid_iou:.4f}')
        writer.add_scalar('validation_loss', valid_loss, epoch)
        writer.add_scalar('validation_dice', valid_dice, epoch)
        writer.add_scalar('validation_iou', valid_iou, epoch)

        # Step the scheduler
        scheduler.step(valid_loss)

    writer.close()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.plot(range(num_epochs), valid_losses, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_dices, label='Train Dice')
    plt.plot(range(num_epochs), valid_dices, label='Valid Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.savefig('loss_dice' + outname + '.png')

    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ortho_dir = "./dataset_fences_france/ortho"
    lidar_dir = "./dataset_fences_france/lidar"
    mask_dir = "./dataset_fences_france/fences_2m"
    gpkg_path = './Departments/departements_grouped.gpkg'
    outname = ''
    batch_size = 16
    num_workers = 4
    num_epochs = 50
    use_lidar = False
    bands = 3
    
    print("[INFO] Training for " + str(outname))

    print("[INFO] Loading data ...")
    train_loader, valid_loader, test_loader = get_data_loaders(ortho_dir, lidar_dir, mask_dir, gpkg_path,  batch_size, num_workers, use_lidar)
    print("[INFO] End loading data.")

    print("[INFO] Creating model ...")
    model = DinkNet101(num_classes=1, n_bands=bands)
    model = model.to(device)

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    print("[INFO] End creating model.")

    print("[INFO] Training ...")
    trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, outname)
    print("[INFO] End training.")

    torch.save(trained_model.state_dict(), outname + '.pth')
