"""Standalone script derived from Advanced_GNN.ipynb cells for syntax checking.
This script is not meant to run fully; it's only to validate Python syntax structure for code in notebook cells.
"""

from __future__ import annotations

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
import torch
import torch.nn.functional as F

# Load datasets

datasets = {}
for name in ["Cora", "CiteSeer", "PubMed"]:
    datasets[name] = Planetoid(root=f"./{name}", name=name)[0]

# Define models


class GCN(torch.nn.Module):
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.c1 = GCNConv(in_ch, hid)
        self.c2 = GCNConv(hid, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.c1(x, edge_index))
        return self.c2(x, edge_index)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.s1 = SAGEConv(in_ch, hid)
        self.s2 = SAGEConv(hid, out_ch)

    def forward(self, x, edge_index):
        x = F.relu(self.s1(x, edge_index))
        return self.s2(x, edge_index)


class GAT(torch.nn.Module):
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.g1 = GATConv(in_ch, hid, heads=4)
        self.g2 = GATConv(4 * hid, out_ch)

    def forward(self, x, edge_index):
        x = F.elu(self.g1(x, edge_index))
        return self.g2(x, edge_index)


# Training helper


def train(model, data, epochs=60):
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()
    return out


# Iterate over dataset and models

results = {}
for ds_name, ds in datasets.items():
    data = ds
    models = {
        "GCN": GCN(data.num_node_features, 32, data.num_classes),
        "GraphSAGE": GraphSAGE(data.num_node_features, 32, data.num_classes),
        "GAT": GAT(data.num_node_features, 8, data.num_classes),
    }

    results[ds_name] = {}

    for name, model in models.items():
        out = train(model, data)
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        results[ds_name][name] = float(acc)

# Over-smoothing demo per dataset
for ds_name, ds in datasets.items():
    print("Dataset:", ds_name)
    x = ds.x.clone()
    edge = ds.edge_index
    conv = GCNConv(ds.num_node_features, ds.num_node_features)

    with torch.no_grad():
        for i in range(8):
            x = conv(x, edge)
            print("Layer", i, "Std:", x.std().item())
    print("---")
