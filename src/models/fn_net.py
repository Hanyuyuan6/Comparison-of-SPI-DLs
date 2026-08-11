import torch.nn as nn
from .base_model import BaseReconstructionNet

class FNReconstructionNet(BaseReconstructionNet):
    """Fully-connected network."""
    def __init__(self, bucket_size, img_size=128,
                 hidden_dims=[1024, 2048, 4096, 8192],
                 dropout_rate=0.2, use_batch_norm=True):
        super().__init__(bucket_size, img_size)

        layers = []
        in_dim = bucket_size

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim

        layers.extend([
            nn.Linear(in_dim, img_size * img_size),
            nn.Sigmoid()
        ])

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, 1, self.img_size, self.img_size)