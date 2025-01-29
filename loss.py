import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch import Unet
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from model import UNet
from datetime import datetime

def get_loss_function(loss_type):
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss(alpha=0.25, gamma=2)
    elif loss_type == 'combined':
        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice(inputs, targets)
        bce_loss = self.bce(torch.sigmoid(inputs), targets)
        return dice_loss + bce_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            # Aplatissement pour segmentation multi-classes
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.permute(0, 2, 1).contiguous().view(-1, inputs.size(1))
        targets = targets.view(-1).long()  # Conversion en type `long`

        # Calcul de la probabilité avec log_softmax pour la stabilité numérique
        log_prob = nn.functional.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)

        # Sélectionne la probabilité correcte pour chaque classe
        log_prob = log_prob[range(len(targets)), targets]
        prob = prob[range(len(targets)), targets]

        # Applique la focal loss
        loss = -self.alpha * (1 - prob) ** self.gamma * log_prob

        # Réduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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