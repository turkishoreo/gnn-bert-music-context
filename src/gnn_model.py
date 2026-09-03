"""
Task 2 (Medium): GNN on music structure graphs (chord-transition or segment graphs).

GraphSAGE update:
  h_i^(l+1) = sigma( W^(l) . CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)) )

Mean-pool readout:
  g = (1/|V|) sum_i h_i^(L),   y_hat = sigmoid(W g + b)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """encoder in {'graphsage', 'gat'}."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, encoder="graphsage",
                 heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for l in range(num_layers):
            d_in = in_dim if l == 0 else hidden_dim
            if encoder == "graphsage":
                self.convs.append(SAGEConv(d_in, hidden_dim))
            elif encoder == "gat":
                # keep output width == hidden_dim regardless of head count
                out_per_head = max(1, hidden_dim // heads)
                self.convs.append(GATConv(d_in, out_per_head, heads=heads, concat=True))
                hidden_dim = out_per_head * heads
            else:
                raise ValueError(f"unknown encoder {encoder}")
        self.out_dim = hidden_dim

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)  # graph-level readout
        return h, g  # node embeddings, graph embedding


class GNNTagClassifier(nn.Module):
    """Task 2 end-to-end: GNN encoder + mean-pool readout + multi-label head,
    matching the spec's 'predict genre or top tags' deliverable."""

    def __init__(self, in_dim, num_tags, hidden_dim=128, num_layers=3,
                 encoder="graphsage", dropout=0.2):
        super().__init__()
        self.gnn = GNNEncoder(in_dim, hidden_dim, num_layers, encoder, dropout=dropout)
        self.head = nn.Linear(self.gnn.out_dim, num_tags)

    def forward(self, x, edge_index, batch):
        _, g = self.gnn(x, edge_index, batch)
        return self.head(g)


class CNNMelBaseline(nn.Module):
    """B2 baseline from spec Section 8: CNN on mel-spectrogram, no graph, no text."""

    def __init__(self, num_tags, n_mels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Linear(32 * 4 * 4, num_tags)

    def forward(self, mel):  # mel: (B, 1, n_mels, frames)
        h = self.conv(mel)
        h = h.flatten(1)
        return self.head(h)
