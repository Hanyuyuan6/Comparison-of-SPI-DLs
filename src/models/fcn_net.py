import torch.nn as nn
from .base_model import BaseReconstructionNet

class FCNReconstructionNet(BaseReconstructionNet):
    def __init__(self, bucket_size, img_size=128, base_channels=1024, fc_mid=7500):
        super().__init__(bucket_size, img_size)
        self.init_size = img_size // 32
        self.init_channels = base_channels
        actual_bucket_size = self.init_channels * self.init_size * self.init_size

        self.input_proj = nn.Sequential(
            nn.Linear(bucket_size, fc_mid),
            nn.LayerNorm(fc_mid),
            nn.ReLU(inplace=True),
            nn.Linear(fc_mid, actual_bucket_size),
            nn.LayerNorm(actual_bucket_size),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.init_channels, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x.view(batch_size, self.init_channels, self.init_size, self.init_size)
        x = self.decoder(x)
        return x