import torch
from torch import nn
import torchtuples as tt
from pycox.models import CoxPH

DEBUG = False

class CustomMLP(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        """
        MLP specifically designed for input features of size 768 and output 1.
        The architecture is: 768 -> 512 -> 128 -> 1.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, out_features)
        )

    def forward(self, x):
        return self.net(x)


class CenteredModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        if DEBUG:
            print(f"[DEBUG] CenteredModel.forward - raw risk scores: mean={out.mean().item():.4f}, "
                  f"std={out.std().item():.4f}, min={out.min().item():.4f}, max={out.max().item():.4f}")
        # Center risk scores for numerical stability
        return out - out.mean(dim=0, keepdim=True)


class CoxPHWithL1(CoxPH):
    """
    Implements the loss:

        L = (1 - α) * L_Cox + α * R(β)

    where the Cox partial likelihood loss is

        L_Cox = (1/|D_T|) ∑_{i∈D_T} δ_i * log( ∑_{j∈S_i} exp(g(x_j) - g(x_i)) )

    and the regularizer is defined as

        R(β) = ((1-γ)/2)*||β||_2^2 + γ*||β||_1

    The hyperparameter α controls the weight of the regularizer, while γ controls the mix of L1 and L2 penalties.
    """
    def __init__(self, net, optimizer, alpha=0.5, gamma=0.5):
        super().__init__(net, optimizer)
        self.alpha = alpha  # Regularization weight.
        self.gamma = gamma  # Relative weight between L1 and L2 regularization.

    def loss(self, preds, durations, events):
        # Compute the standard Cox partial likelihood loss.
        cox_loss = super().loss(preds, durations, events)
        
        # Accumulate L1 and L2 penalties for weights (β) in all linear layers.
        reg_l1 = 0.0
        reg_l2 = 0.0
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                # Only include the weight parameters (ignore biases)
                reg_l1 += torch.sum(torch.abs(module.weight))
                reg_l2 += torch.sum(module.weight ** 2)
                
        # Compute the combined regularizer R(β)
        reg_loss = ((1 - self.gamma) / 2) * reg_l2 + self.gamma * reg_l1
        
        # Combine the Cox loss and regularizer with the specified weights.
        total_loss = (1 - self.alpha) * cox_loss + self.alpha * reg_loss
        return total_loss
