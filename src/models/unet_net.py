import torch
import torch.nn as nn
from .unet_parts import DoubleConv, Down, Up, OutConv
from .base_model import BaseReconstructionNet

class UNetReconstructionNet(BaseReconstructionNet):
    def __init__(self, bucket_size, img_size=128, bilinear=False):
        super().__init__(bucket_size, img_size)
        self.fc = nn.Sequential(
            nn.Linear(bucket_size, img_size * img_size),
            nn.Sigmoid()
        )
        self.inc = DoubleConv(1, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        self.down3 = Down(512, 1024)
        factor = 2 if bilinear else 1
        self.down4 = Down(1024, 2048 // factor)
        self.up1 = Up(2048, 1024 // factor, bilinear)
        self.up2 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, 1)

    def forward(self, x):
        x = self.fc(x).view(-1, 1, self.img_size, self.img_size)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)