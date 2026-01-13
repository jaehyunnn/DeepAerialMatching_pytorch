"""Correlation modules for aerial image matching."""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureCorrelation(nn.Module):
    """Compute correlation between two feature maps using dot product."""
    def __init__(self, temperature: float = 0.07, dual_softmax: bool = True):
        super().__init__()
        self.temperature = temperature
        self.dual_softmax = dual_softmax

    def forward(self, feature_A: torch.Tensor, feature_B: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature_A.size()
        hw = h * w

        # Reshape for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().reshape(b, c, hw)
        feature_B = feature_B.reshape(b, c, hw).transpose(1, 2)

        # Compute correlation
        feature_mul = torch.bmm(feature_B, feature_A)
        # correlation = feature_mul.reshape(b, h, w, hw).permute(0, 3, 1, 2)

        # Dual softmax for scaling correlation features
        if self.dual_softmax:
            feature_mul /= self.temperature
            P = feature_mul.softmax(dim=-1) * feature_mul.softmax(dim=-2) # (B, HW_B, HW_A)
            correlation = P.reshape(b, h, w, hw).permute(0, 3, 1, 2)
        else:
            correlation = feature_mul.reshape(b, h, w, hw).permute(0, 3, 1, 2)

        # MPS has a bug in autograd with non-contiguous tensors during backward pass.
        # Make output contiguous to avoid "view size not compatible" errors.
        return correlation.contiguous()
