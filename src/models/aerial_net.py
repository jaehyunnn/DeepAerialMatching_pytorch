"""Aerial image matching networks."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .backbone import FeatureExtraction
from .correlation import FeatureCorrelation
from .layers import FeatureL2Norm, FeatureRegression


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    """Get device, auto-detecting if not specified."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class AerialNetBase(nn.Module):
    """Base class for aerial image matching networks."""

    # Backbone output channels for feature projection
    BACKBONE_CHANNELS = {
        'vgg': 512,
        'resnet101': 1024,
        'resnext101': 1024,
        'se_resnext101': 1024,
        'densenet169': 1280,
        'vit-l/16': 1024,
    }

    def __init__(
        self,
        geometric_model: str = 'affine',
        normalize_features: bool = True,
        normalize_matches: bool = True,
        device: Optional[torch.device] = None,
        backbone: str = 'vit-l/16',
        freeze_backbone: bool = True,
        version: str | None = None,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches

        # Auto-detect version based on backbone if not specified
        # v1: Legacy CNN backbones (dual_softmax=False, BatchNorm)
        # v2: Modern ViT backbones (dual_softmax=True, GroupNorm)
        if version is None:
            version = 'v2' if backbone.startswith('vit-') else 'v1'
        self.version = version

        # Build model on CPU first
        self.feature_extraction = FeatureExtraction(
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            use_grad_checkpoint=use_grad_checkpoint,
        )
        self.l2_norm = FeatureL2Norm()

        # Version-specific settings
        dual_softmax = (version == 'v2')
        use_batch_norm = (version == 'v1')

        # Setup correlation module
        self.correlation = FeatureCorrelation(dual_softmax=dual_softmax)

        output_dim = 6 if geometric_model == 'affine' else 6
        self.regression = FeatureRegression(output_dim, use_batch_norm=use_batch_norm)
        self.relu = nn.ReLU(inplace=True)

        # Move entire model to device at once
        target_device = _get_device(device)
        self.to(target_device)

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract and optionally normalize features from image."""
        features = self.feature_extraction(image)
        if self.normalize_features:
            features = self.l2_norm(features)
        return features

    def compute_correlation(self, feature_A: torch.Tensor, feature_B: torch.Tensor) -> torch.Tensor:
        """Compute correlation between features and optionally normalize."""
        corr = self.correlation(feature_A, feature_B)

        if self.normalize_matches and not getattr(self.correlation, "dual_softmax", False):
            corr = self.l2_norm(self.relu(corr))
        return corr


class AerialNetSingleStream(AerialNetBase):
    """Single-stream network for aerial image matching.

    Uses bidirectional correlation to estimate affine transformation
    between source and target images.
    """

    def forward(self, tnf_batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # Extract features
        feature_A = self.extract_features(tnf_batch['source_image'])
        feature_B = self.extract_features(tnf_batch['target_image'])

        # Compute bidirectional correlation
        corr_AB = self.compute_correlation(feature_A, feature_B)
        corr_BA = self.compute_correlation(feature_B, feature_A)

        # Regress transformation parameters
        theta_AB = self.regression(corr_AB)
        theta_BA = self.regression(corr_BA)

        return theta_AB, theta_BA


class AerialNetTwoStream(AerialNetBase):
    """Two-stream network with jittered target for robust matching.

    Extends single-stream by adding a jittered target stream for
    improved robustness during training.
    """

    def forward(self, tnf_batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract features from all images
        feature_A = self.extract_features(tnf_batch['source_image'])
        feature_B = self.extract_features(tnf_batch['target_image'])
        feature_C = self.extract_features(tnf_batch['target_image_jit'])

        # Compute bidirectional correlation for original pair
        corr_AB = self.compute_correlation(feature_A, feature_B)
        corr_BA = self.compute_correlation(feature_B, feature_A)

        # Compute bidirectional correlation for jittered pair
        corr_AC = self.compute_correlation(feature_A, feature_C)
        corr_CA = self.compute_correlation(feature_C, feature_A)

        # Regress transformation parameters
        theta_AB = self.regression(corr_AB)
        theta_BA = self.regression(corr_BA)
        theta_AC = self.regression(corr_AC)
        theta_CA = self.regression(corr_CA)

        return theta_AB, theta_BA, theta_AC, theta_CA
