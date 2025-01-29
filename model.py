import torch
import torch.nn as nn
import torch.optim as optim
from segmentation_models_pytorch import Unet
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_data_loaders  # Ensure this imports your data loader script
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        """
        Convolution suivie d'une option de normalisation par lots et d'une activation ReLU.

        Args:
            in_channels (int): Nombre de canaux d'entrée.
            out_channels (int): Nombre de canaux de sortie.
            kernel_size (int or tuple): Taille du noyau de convolution.
            padding (int or tuple, optional): Remplissage (padding) de la convolution. Par défaut 0.
            stride (int or tuple, optional): Pas (stride) de la convolution. Par défaut 1.
            use_batchnorm (bool, optional): Si True, ajoute une normalisation par lots après la convolution. Par défaut True.
        """
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity() 

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # Upsampling
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # Skip connection
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetLateFusion(nn.Module):
    def __init__(self, decoder_channels, n_classes=1, use_batchnorm=True):
        super(UNetLateFusion, self).__init__()

        # ----------------------- UNet for RGB--------------------------
        # Encoder
        self.rgb_e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.rgb_e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.rgb_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rgb_e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.rgb_e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.rgb_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rgb_e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.rgb_e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.rgb_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rgb_e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.rgb_e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.rgb_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rgb_e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.rgb_e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        encoder_channels = [1024, 512, 256, 128, 64] 
        kwargs = dict(use_batchnorm=use_batchnorm)
        self.decoder_blocks_rgb = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(
                encoder_channels[:-1],  
                encoder_channels[1:], 
                decoder_channels      
            )
        ])

        # ----------------------- UNet for DSM--------------------------
        # Encoder
        self.dsm_e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.dsm_e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dsm_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dsm_e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dsm_e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dsm_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dsm_e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dsm_e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dsm_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dsm_e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.dsm_e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dsm_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dsm_e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.dsm_e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        encoder_channels = [1024, 512, 256, 128, 64] 
        kwargs = dict(use_batchnorm=use_batchnorm)
        self.decoder_blocks_dsm = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(
                encoder_channels[:-1],  
                encoder_channels[1:],  
                decoder_channels       
            )
        ])

        # Output layer for binary segmentation
        self.outconv = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, img, dsm):
        # ----------------------- UNet for RGB--------------------------
        # Encoder
        rgb_xe11 = F.relu(self.rgb_e11(img))
        rgb_xe12 = F.relu(self.rgb_e12(rgb_xe11))
        rgb_xp1 = self.rgb_pool1(rgb_xe12)

        rgb_xe21 = F.relu(self.rgb_e21(rgb_xp1))
        rgb_xe22 = F.relu(self.rgb_e22(rgb_xe21))
        rgb_xp2 = self.rgb_pool2(rgb_xe22)

        rgb_xe31 = F.relu(self.rgb_e31(rgb_xp2))
        rgb_xe32 = F.relu(self.rgb_e32(rgb_xe31))
        rgb_xp3 = self.rgb_pool3(rgb_xe32)

        rgb_xe41 = F.relu(self.rgb_e41(rgb_xp3))
        rgb_xe42 = F.relu(self.rgb_e42(rgb_xe41))
        rgb_xp4 = self.rgb_pool4(rgb_xe42)

        rgb_xe51 = F.relu(self.rgb_e51(rgb_xp4))
        rgb_xe52 = F.relu(self.rgb_e52(rgb_xe51))

        features_rgb = [rgb_xe52, rgb_xe42, rgb_xe32, rgb_xe22, rgb_xe12]  # Features from encoder
        x_rgb = features_rgb[0]

        for i, decoder_block in enumerate(self.decoder_blocks_rgb):
            skip = features_rgb[i + 1] if i + 1 < len(features_rgb) else None
            x_rgb = decoder_block(x_rgb, skip)

        # ----------------------- UNet for DSM--------------------------
        # Encoder
        dsm_xe11 = F.relu(self.dsm_e11(dsm))
        dsm_xe12 = F.relu(self.dsm_e12(dsm_xe11))
        dsm_xp1 = self.dsm_pool1(dsm_xe12)

        dsm_xe21 = F.relu(self.dsm_e21(dsm_xp1))
        dsm_xe22 = F.relu(self.dsm_e22(dsm_xe21))
        dsm_xp2 = self.dsm_pool2(dsm_xe22)

        dsm_xe31 = F.relu(self.dsm_e31(dsm_xp2))
        dsm_xe32 = F.relu(self.dsm_e32(dsm_xe31))
        dsm_xp3 = self.dsm_pool3(dsm_xe32)

        dsm_xe41 = F.relu(self.dsm_e41(dsm_xp3))
        dsm_xe42 = F.relu(self.dsm_e42(dsm_xe41))
        dsm_xp4 = self.dsm_pool4(dsm_xe42)

        dsm_xe51 = F.relu(self.dsm_e51(dsm_xp4))
        dsm_xe52 = F.relu(self.dsm_e52(dsm_xe51))

        # Decoder forward pass
        features = [dsm_xe52, dsm_xe42, dsm_xe32, dsm_xe22, dsm_xe12]  # Features from encoder
        x_dsm = features[0]

        for i, decoder_block in enumerate(self.decoder_blocks_dsm):
            skip = features[i + 1] if i + 1 < len(features) else None
            x_dsm = decoder_block(x_dsm, skip)

        # Fusion before final activation function
        final_cat = torch.cat([x_dsm, x_rgb], dim=1)

        # Output layer
        out = self.outconv(final_cat)
        return out


class UNet(nn.Module):
    def __init__(self, input_channels, decoder_channels, n_classes=1, use_batchnorm=True):
        super().__init__()

        # Encoder
        self.e11 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        encoder_channels = [1024, 512, 256, 128, 64]  # Correspond aux sorties des encodeurs
        kwargs = dict(use_batchnorm=use_batchnorm)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(
                encoder_channels[:-1],  # Les in_channels proviennent de l'encodeur
                encoder_channels[1:],  # Les skip_channels correspondent aux connexions de saut
                decoder_channels       # Les out_channels sont définis dans decoder_channels
            )
        ])

        # Output layer
        self.outconv = nn.Conv2d(decoder_channels[-1], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        # Decoder forward pass
        features = [xe52, xe42, xe32, xe22, xe12]  # Features from encoder
        x = features[0]

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = features[i + 1] if i + 1 < len(features) else None
            x = decoder_block(x, skip)

        # Output layer
        out = self.outconv(x)
        return out


class UNetFeatFusion(nn.Module):
    def __init__(self, n_class=1):
        super(UNetFeatFusion, self).__init__()
        # RGB is the main network, working with the features concatenation

        # RGB Encoder - 1
        self.rgb_e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.rgb_e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.rgb_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Depth Encoder - 1
        self.depth_e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.depth_e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.depth_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RGB Encoder - 2
        self.rgb_e21 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.rgb_e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.rgb_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Depth Encoder - 2
        self.depth_e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.depth_e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.depth_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RGB Encoder - 3
        self.rgb_e31 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.rgb_e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.rgb_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Depth Encoder - 3
        self.depth_e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.depth_e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.depth_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RGB Encoder - 4
        self.rgb_e41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.rgb_e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.rgb_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Depth Encoder - 4
        self.depth_e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.depth_e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.depth_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.e51 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, img, dsm):
        # RGB and Depth Encoder - step one
        rgb_xe11 = F.relu(self.rgb_e11(img))
        rgb_xe12 = F.relu(self.rgb_e12(rgb_xe11))
        rgb_xp1 = self.rgb_pool1(rgb_xe12)

        depth_xe11 = F.relu(self.depth_e11(dsm))
        depth_xe12 = F.relu(self.depth_e12(depth_xe11))
        depth_xp1 = self.depth_pool1(depth_xe12)

        # Fusion step
        xp1 = torch.cat([rgb_xp1, depth_xp1], dim=1)

        # RGB and Depth Encoder - step two
        rgb_xe21 = F.relu(self.rgb_e21(xp1)) #Here we take the concatenation from above, RGB is the main network
        rgb_xe22 = F.relu(self.rgb_e22(rgb_xe21))
        rgb_xp2 = self.rgb_pool2(rgb_xe22)

        depth_xe21 = F.relu(self.depth_e21(depth_xp1))
        depth_xe22 = F.relu(self.depth_e22(depth_xe21))
        depth_xp2 = self.depth_pool2(depth_xe22)

        # Fusion step
        xp2 = torch.cat([rgb_xp2, depth_xp2], dim=1)

        # RGB and Depth Encoder - step three
        rgb_xe31 = F.relu(self.rgb_e31(xp2))
        rgb_xe32 = F.relu(self.rgb_e32(rgb_xe31))
        rgb_xp3 = self.rgb_pool3(rgb_xe32)

        depth_xe31 = F.relu(self.depth_e31(depth_xp2))
        depth_xe32 = F.relu(self.depth_e32(depth_xe31))
        depth_xp3 = self.depth_pool3(depth_xe32)

        # Fusion step
        xp3 = torch.cat([rgb_xp3, depth_xp3], dim=1)

        # RGB and Depth Encoder - step four
        rgb_xe41 = F.relu(self.rgb_e41(xp3))
        rgb_xe42 = F.relu(self.rgb_e42(rgb_xe41))
        rgb_xp4 = self.rgb_pool4(rgb_xe42)

        depth_xe41 = F.relu(self.depth_e41(depth_xp3))
        depth_xe42 = F.relu(self.depth_e42(depth_xe41))
        depth_xp4 = self.depth_pool4(depth_xe42)

        # Fusion step
        xp4 = torch.cat([rgb_xp4, depth_xp4], dim=1)

        # Bottleneck
        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, rgb_xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, rgb_xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, rgb_xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, rgb_xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)
        return out

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