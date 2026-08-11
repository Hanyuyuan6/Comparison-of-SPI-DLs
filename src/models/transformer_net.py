import torch
import torch.nn as nn
import math
from .base_model import BaseReconstructionNet

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerReconstructionNet(BaseReconstructionNet):
    def __init__(self, bucket_size, img_size=128, d_model=512, nhead=8, num_layers=6):
        super().__init__(bucket_size, img_size)
        self.d_model = d_model
        self.seq_len = min(32, bucket_size)
        while bucket_size % self.seq_len != 0 and self.seq_len > 1:
            self.seq_len -= 1
        self.patch_size = bucket_size // self.seq_len

        self.input_embedding = nn.Linear(self.patch_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, self.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.init_feat_hw = 16
        self.init_feat_ch = 32

        self.mlp_to_feat = nn.Sequential(
            nn.Linear(d_model * self.seq_len, self.init_feat_ch * self.init_feat_hw * self.init_feat_hw),
            nn.LayerNorm(self.init_feat_ch * self.init_feat_hw * self.init_feat_hw),
            nn.ReLU(inplace=True)
        )

        self.decoder = self._build_decoder(self.init_feat_ch)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.patch_size)
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.contiguous().view(batch_size, -1)
        feat = self.mlp_to_feat(x)
        feat = feat.view(batch_size, self.init_feat_ch, self.init_feat_hw, self.init_feat_hw)
        out = self.decoder(feat)
        return out