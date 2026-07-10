import torch
import torch.nn as nn


class UncertaintyHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=16):
        super().__init__()
        self.dropout = nn.Dropout2d(p=0.3)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


class DualUncertaintyHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=16):
        super().__init__()
        self.head1 = UncertaintyHead(in_channels, hidden_channels)
        self.head2 = UncertaintyHead(in_channels, hidden_channels)

    def forward(self, x):
        u1 = self.head1(x)
        u2 = self.head2(x)
        disagreement = torch.var(torch.stack([u1, u2], dim=0), dim=0)
        mean_uncertainty = (u1 + u2) / 2
        return mean_uncertainty, disagreement
