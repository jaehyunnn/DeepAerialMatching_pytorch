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

    def __init__(self, output_dim: int = 6, use_cuda: bool = True, add_coord: bool = False):
        super().__init__()
        self.add_coord = add_coord

        # Input channels: 15*15 correlation + 2 coord channels if add_coord
        in_channels = 15 * 15 + 2 if add_coord else 15 * 15

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 5 * 5, output_dim)

        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add_coord:
            B, _, H, W = x.shape

            # Normalized coordinates (-1 ~ 1)
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
                torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
                indexing='ij'
            )
            grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

            # Concatenate features with coordinates
            x = torch.cat([x, grid], dim=1)

        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x
