import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseReconstructionNet

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output + self.bias

class GCNReconstructionNet(BaseReconstructionNet):
    def __init__(self, bucket_size, img_size=128, hidden_channels=256, num_layers=8):
        super().__init__(bucket_size, img_size)
        self.num_nodes = 32
        while bucket_size % self.num_nodes != 0 and self.num_nodes > 1:
            self.num_nodes -= 1
        self.node_features = bucket_size // self.num_nodes

        self.input_proj = nn.Linear(self.node_features, hidden_channels)

        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.gcn_layers.append(GraphConv(hidden_channels, hidden_channels))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.gcn_layers.append(GraphConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

        self.init_feat_hw = 16
        self.init_feat_ch = 32

        self.mlp_to_feat = nn.Sequential(
            nn.Linear(hidden_channels * self.num_nodes, self.init_feat_ch * self.init_feat_hw * self.init_feat_hw),
            nn.LayerNorm(self.init_feat_ch * self.init_feat_hw * self.init_feat_hw),
            nn.ReLU(inplace=True)
        )

        self.decoder = self._build_decoder(self.init_feat_ch)

        self.adj_weight = nn.Parameter(torch.randn(self.num_nodes, self.num_nodes))

    def get_adjacency_matrix(self, batch_size):
        adj = torch.sigmoid(self.adj_weight)
        adj = (adj + adj.t()) / 2
        adj = adj + torch.eye(self.num_nodes, device=adj.device)
        row_sum = adj.sum(1)
        adj_normalized = adj / row_sum.unsqueeze(1)
        return adj_normalized.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_nodes, self.node_features)
        x = self.input_proj(x)

        adj = self.get_adjacency_matrix(batch_size)

        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            x = gcn(x, adj)
            x_bn = x.view(-1, x.size(-1))
            x_bn = bn(x_bn)
            x = x_bn.view(batch_size, self.num_nodes, -1)
            x = F.relu(x)
            if i < len(self.gcn_layers) - 1:
                x = F.dropout(x, p=0.2, training=self.training)

        x = x.view(batch_size, -1)
        feat = self.mlp_to_feat(x)
        feat = feat.view(batch_size, self.init_feat_ch, self.init_feat_hw, self.init_feat_hw)
        out = self.decoder(feat)
        return out