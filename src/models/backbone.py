"""Feature extraction backbones for aerial image matching."""
from __future__ import annotations

import torch.nn as nn
import timm


class FeatureExtraction(nn.Module):
    """Feature extraction using timm backbones.

    Supports CNN backbones (ResNet, DenseNet, etc.) and ViT backbones (DINOv3).
    All backbones produce 15x15 feature maps for 240x240 input images.
    """

    # Supported backbones and their timm model names
    BACKBONES = {
        'vgg': 'vgg16',
        'resnet101': 'resnet101',
        'resnext101': 'resnext101_32x4d',
        'se_resnext101': 'seresnext101_32x4d',
        'densenet169': 'densenet169',
        'dinov3': 'vit_large_patch16_dinov3',
    }

    def __init__(
        self,
        backbone: str = 'se_resnext101',
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Supported: {list(self.BACKBONES.keys())}")

        self.backbone = backbone
        self.is_vit = backbone.startswith('dino')

        if self.is_vit:
            self._init_vit_backbone(backbone)
        else:
            self._init_cnn_backbone(backbone)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    def _init_cnn_backbone(self, backbone: str):
        """Initialize CNN backbone with appropriate layer selection."""
        model_name = self.BACKBONES[backbone]
        full_model = timm.create_model(model_name, pretrained=True)

        if backbone == 'vgg':
            # Extract features up to pool4 (15x15 output for 240x240 input)
            self.model = nn.Sequential(*list(full_model.features.children())[:24])

        elif backbone in ('resnet101', 'resnext101', 'se_resnext101'):
            # Extract up to layer3 (15x15 output for 240x240 input)
            self.model = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.act1,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2,
                full_model.layer3,
            )

        elif backbone == 'densenet169':
            # Extract up to denseblock3 (15x15 output for 240x240 input)
            self.model = nn.Sequential(*list(full_model.features.children())[:8])

    def _init_vit_backbone(self, backbone: str):
        """Initialize ViT backbone for spatial feature extraction.

        For 240x240 input with patch_size=16:
        - num_patches = (240/16) * (240/16) = 15 * 15 = 225
        - Output is reshaped to (B, embed_dim, 15, 15)
        """
        model_name = self.BACKBONES[backbone]
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            img_size=240,
            num_classes=0,  # Remove classification head
        )
        self.patch_size = 16
        self.embed_dim = self.model.embed_dim

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        if self.is_vit:
            return self._forward_vit(image_batch)
        return self.model(image_batch)

    def _forward_vit(self, image_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for ViT backbone.

        Returns spatial feature map of shape (B, embed_dim, H, W).
        """
        B = image_batch.size(0)
        H = W = image_batch.size(2) // self.patch_size  # 240 // 16 = 15

        # Get patch embeddings (excluding CLS token if present)
        x = self.model.forward_features(image_batch)

        # Handle CLS token: some models have it, some don't
        if hasattr(self.model, 'num_prefix_tokens') and self.model.num_prefix_tokens > 0:
            x = x[:, self.model.num_prefix_tokens:]  # Remove CLS/register tokens

        # Reshape to spatial feature map: (B, H*W, C) -> (B, C, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x
