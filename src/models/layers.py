"""Common neural network layers for aerial image matching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureL2Norm(nn.Module):
    """L2 normalization of features along channel dimension."""

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(feature, p=2, dim=1, eps=1e-6)


class FeatureRegression(nn.Module):
    """Regress transformation parameters from correlation tensor."""

    def __init__(
        self,
        output_dim: int = 6,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        # Input channels: 15*15 correlation
        in_channels = 15 * 15

        # v1: BatchNorm2d (legacy CNN backbones), v2: GroupNorm (modern, stable for small batches)
        if use_batch_norm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = lambda c: nn.GroupNorm(8, c)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=0),
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            norm_layer(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 5 * 5, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x
