"""Correlation modules for aerial image matching."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureCorrelation(nn.Module):
    """Compute correlation between two feature maps using dot product."""

    def forward(self, feature_A: torch.Tensor, feature_B: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feature_A.size()

        # Reshape for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().reshape(b, c, h * w)
        feature_B = feature_B.reshape(b, c, h * w).transpose(1, 2)

        # Compute correlation
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation = feature_mul.reshape(b, h, w, h * w).transpose(2, 3).transpose(1, 2)

        # MPS has a bug in autograd with non-contiguous tensors during backward pass.
        # Make output contiguous to avoid "view size not compatible" errors.
        return correlation.contiguous()


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding, flexible to input size (LoFTR-style)."""

    def __init__(self, d_model: int, max_shape: tuple[int, int] = (256, 256)):
        super().__init__()
        self.d_model = d_model

        # Pre-compute positional encodings for max size
        pe = self._make_pe(max_shape[0], max_shape[1], d_model)
        self.register_buffer('pe', pe, persistent=False)

    def _make_pe(self, h: int, w: int, d_model: int) -> torch.Tensor:
        """Generate 2D sinusoidal positional encoding."""
        pe = torch.zeros(d_model, h, w)
        d_model_half = d_model // 2

        # Y-axis encoding
        y_pos = torch.arange(0, h).unsqueeze(1).repeat(1, w).float()
        # X-axis encoding
        x_pos = torch.arange(0, w).unsqueeze(0).repeat(h, 1).float()

        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))

        # Apply sin/cos to y positions (first half of channels)
        pe[0:d_model_half:2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[1:d_model_half:2, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))

        # Apply sin/cos to x positions (second half of channels)
        pe[d_model_half::2, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[d_model_half + 1::2, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: (B, C, H, W) feature map

        Returns:
            (B, C, H, W) feature map with positional encoding added
        """
        _, _, h, w = x.shape
        return x + self.pe[:, :h, :w].unsqueeze(0)


class CrossAttentionLayer(nn.Module):
    """Multi-head cross-attention layer (LoFTR-style)."""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Cross-attention forward pass.

        Args:
            query: (B, N, C) query features
            key: (B, M, C) key features
            value: (B, M, C) value features

        Returns:
            (B, N, C) attended features
        """
        b, n, _ = query.shape
        m = key.shape[1]

        # Project and reshape to multi-head
        q = self.q_proj(query).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = (attn @ v).transpose(1, 2).reshape(b, n, self.d_model)
        return self.out_proj(out)


class CrossAttentionCorrelation(nn.Module):
    """LoFTR-style cross-attention correlation.

    Uses transformer cross-attention with 2D positional encoding
    to compute correlation between two feature maps.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model)

        # Cross-attention layers (interleaved self + cross attention)
        self.self_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])

        # Layer norms
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, feature_A: torch.Tensor, feature_B: torch.Tensor) -> torch.Tensor:
        """Compute cross-attention based correlation.

        Args:
            feature_A: (B, C, H, W) source features
            feature_B: (B, C, H, W) target features

        Returns:
            (B, H*W, H, W) correlation tensor (compatible with FeatureRegression)
        """
        b, _, h, w = feature_A.shape

        # Add positional encoding
        feat_A = self.pos_encoding(feature_A)
        feat_B = self.pos_encoding(feature_B)

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        feat_A = feat_A.flatten(2).transpose(1, 2)
        feat_B = feat_B.flatten(2).transpose(1, 2)

        # Apply interleaved self + cross attention
        for i in range(len(self.self_attn_layers)):
            # Self-attention
            feat_A = feat_A + self.self_attn_layers[i](feat_A, feat_A, feat_A)
            feat_A = self.norm1[i](feat_A)
            feat_B = feat_B + self.self_attn_layers[i](feat_B, feat_B, feat_B)
            feat_B = self.norm1[i](feat_B)

            # Cross-attention
            feat_A = feat_A + self.cross_attn_layers[i](feat_A, feat_B, feat_B)
            feat_A = self.norm2[i](feat_A)
            feat_B = feat_B + self.cross_attn_layers[i](feat_B, feat_A, feat_A)
            feat_B = self.norm2[i](feat_B)

        # Compute correlation: (B, H*W, C) x (B, C, H*W) -> (B, H*W, H*W)
        feat_A = F.normalize(feat_A, p=2, dim=-1)
        feat_B = F.normalize(feat_B, p=2, dim=-1)
        correlation = torch.bmm(feat_A, feat_B.transpose(1, 2))

        # Reshape to (B, H*W, H, W) for compatibility with FeatureRegression
        correlation = correlation.reshape(b, h * w, h, w)

        return correlation
