import torch.nn as nn
from .base_model import BaseReconstructionNet

class LSTMReconstructionNet(BaseReconstructionNet):
    def __init__(self, bucket_size, img_size=128, hidden_size=256, num_layers=3):
        super().__init__(bucket_size, img_size)
        self.seq_len = min(32, bucket_size)
        while bucket_size % self.seq_len != 0 and self.seq_len > 1:
            self.seq_len -= 1
        self.input_size = bucket_size // self.seq_len

        self.input_proj = nn.Linear(self.input_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )

        self.init_feat_hw = 16
        self.init_feat_ch = 32

        self.mlp_to_feat = nn.Sequential(
            nn.Linear(hidden_size * 2 * self.seq_len, self.init_feat_ch * self.init_feat_hw * self.init_feat_hw),
            nn.LayerNorm(self.init_feat_ch * self.init_feat_hw * self.init_feat_hw),
            nn.ReLU(inplace=True)
        )

        self.decoder = self._build_decoder(self.init_feat_ch)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.input_size)
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = x.contiguous().view(batch_size, -1)
        feat = self.mlp_to_feat(x)
        feat = feat.view(batch_size, self.init_feat_ch, self.init_feat_hw, self.init_feat_hw)
        out = self.decoder(feat)
        return out