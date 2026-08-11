import torch.nn as nn
from abc import ABC, abstractmethod

class BaseReconstructionNet(nn.Module, ABC):
    """Base class for reconstruction networks providing common utilities."""
    def __init__(self, bucket_size, img_size=128):
        super().__init__()
        self.bucket_size = bucket_size
        self.img_size = img_size

    @abstractmethod
    def forward(self, x):
        pass

    def _build_decoder(self, in_channels=32):
        """Shared 4-stage transpose-conv decoder used by the sequence/graph backbones:
        in_channels -> 16 -> 8 -> 4 -> 1, 8x upsampling (16x16 feature map -> 128x128 image).
        Assigned to ``self.decoder`` in each backbone so the state_dict keys are identical.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 4, 4, 2, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, 3, 1, 1),
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )