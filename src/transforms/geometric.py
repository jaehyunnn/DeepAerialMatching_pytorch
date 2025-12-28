"""Geometric transformation utilities for image warping."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    """Get device, auto-detecting if not specified."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class AffineGridGen(nn.Module):
    """Generate affine transformation grid for grid_sample.

    Args:
        out_h: Output height.
        out_w: Output width.
    """

    def __init__(self, out_h: int = 240, out_w: int = 240):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Generate affine grid.

        Args:
            theta: Affine transformation matrix (B, 2, 3).

        Returns:
            Sampling grid (B, H, W, 2).
        """
        batch_size = theta.size(0)
        out_size = torch.Size((batch_size, 1, self.out_h, self.out_w))
        return F.affine_grid(theta.contiguous(), out_size, align_corners=False)


class GeometricTnf:
    """Geometric transformation for image batches.

    Applies affine transformations with optional resizing.

    Args:
        geometric_model: Type of geometric model ('affine').
        out_h: Output height.
        out_w: Output width.
        device: Device for computation. If None, auto-detects (CUDA > MPS > CPU).
    """

    def __init__(
        self,
        geometric_model: str = 'affine',
        out_h: int = 240,
        out_w: int = 240,
        device: Optional[torch.device] = None,
    ):
        self.out_h = out_h
        self.out_w = out_w
        self.device = _get_device(device)

        if geometric_model == 'affine':
            self.grid_gen = AffineGridGen(out_h, out_w)

        # Identity transformation matrix
        self.theta_identity = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            device=self.device,
        )

    def __call__(
        self,
        image_batch: torch.Tensor,
        theta_batch: torch.Tensor | None = None,
        padding_factor: float = 1.0,
        crop_factor: float = 1.0,
    ) -> torch.Tensor:
        """Apply geometric transformation to image batch.

        Args:
            image_batch: Input images (B, C, H, W).
            theta_batch: Transformation matrices (B, 2, 3). If None, uses identity.
            padding_factor: Padding scale factor.
            crop_factor: Crop scale factor.

        Returns:
            Warped images (B, C, out_h, out_w).
        """
        batch_size = image_batch.size(0)
        original_device = image_batch.device

        if theta_batch is None:
            theta_batch = self.theta_identity.expand(batch_size, 2, 3).contiguous().to(image_batch.device)

        # MPS has an intermittent bug with grid_sample that corrupts color channels
        # under certain conditions (after model inference, specific image sizes).
        # Run on CPU for correctness, then move back to original device.
        # See: https://github.com/pytorch/pytorch/issues/141287
        if original_device.type == 'mps':
            image_batch = image_batch.cpu()
            theta_batch = theta_batch.cpu()

        sampling_grid = self.grid_gen(theta_batch)
        sampling_grid = sampling_grid * padding_factor * crop_factor

        result = F.grid_sample(image_batch, sampling_grid, align_corners=False)

        if original_device.type == 'mps':
            result = result.to(original_device)

        return result


def symmetric_image_pad(image_batch: torch.Tensor, padding_factor: float) -> torch.Tensor:
    """Apply symmetric (reflection) padding to image batch.

    Args:
        image_batch: Input images (B, C, H, W).
        padding_factor: Fraction of image size to pad.

    Returns:
        Padded images.
    """
    _, _, h, w = image_batch.size()
    pad_h = int(h * padding_factor)
    pad_w = int(w * padding_factor)
    device = image_batch.device

    # Create reflection indices
    idx_left = torch.arange(pad_w - 1, -1, -1, device=device)
    idx_right = torch.arange(w - 1, w - pad_w - 1, -1, device=device)
    idx_top = torch.arange(pad_h - 1, -1, -1, device=device)
    idx_bottom = torch.arange(h - 1, h - pad_h - 1, -1, device=device)

    # Apply horizontal padding
    image_batch = torch.cat([
        image_batch.index_select(3, idx_left),
        image_batch,
        image_batch.index_select(3, idx_right),
    ], dim=3)

    # Apply vertical padding
    image_batch = torch.cat([
        image_batch.index_select(2, idx_top),
        image_batch,
        image_batch.index_select(2, idx_bottom),
    ], dim=2)

    return image_batch


class SynthPairTnf:
    """Generate synthetically warped training pairs.

    Creates source-target pairs with geometric transformations for training.

    Args:
        device: Device for computation. If None, auto-detects (CUDA > MPS > CPU).
        geometric_model: Type of geometric model.
        crop_factor: Crop scale factor.
        output_size: Output image size (H, W).
        padding_factor: Padding scale factor.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        geometric_model: str = 'affine',
        crop_factor: float = 9 / 16,
        output_size: tuple[int, int] = (240, 240),
        padding_factor: float = 0.5,
    ):
        self.device = _get_device(device)
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size

        self.rescaling_tnf = GeometricTnf('affine', self.out_h, self.out_w, device=self.device)
        self.geometric_tnf = GeometricTnf(geometric_model, self.out_h, self.out_w, device=self.device)

    def __call__(self, batch: dict) -> dict:
        """Apply synthetic transformation to batch.

        Args:
            batch: Dict with 'src_image', 'trg_image', 'trg_image_jit', 'theta'.

        Returns:
            Dict with 'source_image', 'target_image', 'target_image_jit', 'theta_GT'.
        """
        src_image = batch['src_image'].to(self.device)
        trg_image = batch['trg_image'].to(self.device)
        trg_image_jit = batch['trg_image_jit'].to(self.device)
        theta = batch['theta'].to(self.device)

        # Apply symmetric padding
        src_image = symmetric_image_pad(src_image, self.padding_factor)
        trg_image = symmetric_image_pad(trg_image, self.padding_factor)
        trg_image_jit = symmetric_image_pad(trg_image_jit, self.padding_factor)

        # Crop source and warp targets
        source_image = self.rescaling_tnf(src_image, None, self.padding_factor, self.crop_factor)
        target_image = self.geometric_tnf(trg_image, theta, self.padding_factor, self.crop_factor)
        target_image_jit = self.geometric_tnf(trg_image_jit, theta, self.padding_factor, self.crop_factor)

        return {
            'source_image': source_image,
            'target_image': target_image,
            'target_image_jit': target_image_jit,
            'theta_GT': theta,
        }


class SynthPairTnfPCK:
    """Generate synthetically warped pairs for PCK evaluation.

    Similar to SynthPairTnf but without jittered target.

    Args:
        device: Device for computation. If None, auto-detects (CUDA > MPS > CPU).
        geometric_model: Type of geometric model.
        crop_factor: Crop scale factor.
        output_size: Output image size (H, W).
        padding_factor: Padding scale factor.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        geometric_model: str = 'affine',
        crop_factor: float = 9 / 16,
        output_size: tuple[int, int] = (240, 240),
        padding_factor: float = 0.5,
    ):
        self.device = _get_device(device)
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size

        self.rescaling_tnf = GeometricTnf('affine', self.out_h, self.out_w, device=self.device)
        self.geometric_tnf = GeometricTnf(geometric_model, self.out_h, self.out_w, device=self.device)

    def __call__(self, batch: dict) -> dict:
        """Apply synthetic transformation to batch.

        Args:
            batch: Dict with 'src_image', 'trg_image', 'theta'.

        Returns:
            Dict with 'source_image', 'target_image'.
        """
        src_image = batch['src_image']
        trg_image = batch['trg_image']
        theta = batch['theta']

        # Apply symmetric padding
        src_image = symmetric_image_pad(src_image, self.padding_factor)
        trg_image = symmetric_image_pad(trg_image, self.padding_factor)

        # Crop source and warp target
        source_image = self.rescaling_tnf(src_image, None, self.padding_factor, self.crop_factor)
        target_image = self.geometric_tnf(trg_image, theta, self.padding_factor, self.crop_factor)

        return {
            'source_image': source_image,
            'target_image': target_image,
        }


def theta2homogeneous(theta: torch.Tensor) -> torch.Tensor:
    """Convert 2x3 affine matrix to 3x3 homogeneous matrix.

    Args:
        theta: Affine matrix (B, 6) or (B, 2, 3).

    Returns:
        Homogeneous matrix (B, 3, 3).
    """
    batch_size = theta.size(0)
    theta = theta.reshape(-1, 2, 3)

    homogeneous_row = torch.tensor(
        [[0.0, 0.0, 1.0]],
        dtype=theta.dtype,
        device=theta.device,
    ).expand(batch_size, 1, 3).contiguous()

    return torch.cat([theta, homogeneous_row], dim=1)
