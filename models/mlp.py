# models/mlp.py
import torch
from torch import nn
import torchtuples as tt
from pycox.models import CoxPH

DEBUG = False

class CustomMLP(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, dropout=0.0, l1_lambda=0.0):
        super().__init__()
        self.l1_lambda = l1_lambda

        layers = []
        prev_features = in_features
        for nodes in num_nodes:
            layers.append(nn.Linear(prev_features, nodes))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_features = nodes

        layers.append(nn.Linear(prev_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def l1_regularization(self):
        l1_loss = 0.0
        for module in self.net:
            if isinstance(module, nn.Linear):
                l1_loss += torch.sum(torch.abs(module.weight))
        return self.l1_lambda * l1_loss

class CenteredModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        if DEBUG:
            print(f"[DEBUG] CenteredModel.forward - raw risk scores: mean={out.mean().item():.4f}, "
                  f"std={out.std().item():.4f}, min={out.min().item():.4f}, max={out.max().item():.4f}")
        return out - out.mean(dim=0, keepdim=True)

class CoxPHWithL1(CoxPH):
    def loss(self, preds, durations, events):
        base_loss = super().loss(preds, durations, events)
        reg_loss = self.net.l1_regularization() if hasattr(self.net, 'l1_regularization') else 0.0
        return base_loss + reg_loss
