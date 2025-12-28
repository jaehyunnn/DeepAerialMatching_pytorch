"""Point transformation utilities for coordinate manipulation."""
from __future__ import annotations

import torch


class PointTnf:
    """Transform points with affine transformations."""

    def __init__(self):
        pass

    def affine_transform(self, theta: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation to points.

        Args:
            theta: Affine matrix (B, 6) or (B, 2, 3).
            points: Points to transform (B, 2, N).

        Returns:
            Transformed points (B, 2, N).
        """
        theta_mat = theta.reshape(-1, 2, 3)
        # Apply rotation/scale
        warped_points = torch.bmm(theta_mat[:, :, :2], points)
        # Apply translation
        warped_points = warped_points + theta_mat[:, :, 2:3]
        return warped_points


def points_to_unit_coords(points: torch.Tensor, im_size: torch.Tensor) -> torch.Tensor:
    """Convert pixel coordinates to normalized coordinates (-1 to 1).

    Args:
        points: Points in pixel coordinates (B, 2, N).
        im_size: Image size (B, 2) where [:, 0] is height and [:, 1] is width.

    Returns:
        Points in normalized coordinates (B, 2, N).
    """
    h = im_size[:, 0:1].float()  # (B, 1)
    w = im_size[:, 1:2].float()  # (B, 1)

    points_norm = points.clone()
    # Normalize X (dim 0): pixel -> [-1, 1]
    points_norm[:, 0, :] = (points[:, 0, :] - 1 - (w - 1) / 2) * 2 / (w - 1)
    # Normalize Y (dim 1): pixel -> [-1, 1]
    points_norm[:, 1, :] = (points[:, 1, :] - 1 - (h - 1) / 2) * 2 / (h - 1)

    return points_norm


def points_to_pixel_coords(points: torch.Tensor, im_size: torch.Tensor) -> torch.Tensor:
    """Convert normalized coordinates (-1 to 1) to pixel coordinates.

    Args:
        points: Points in normalized coordinates (B, 2, N).
        im_size: Image size (B, 2) where [:, 0] is height and [:, 1] is width.

    Returns:
        Points in pixel coordinates (B, 2, N).
    """
    h = im_size[:, 0:1].float()  # (B, 1)
    w = im_size[:, 1:2].float()  # (B, 1)

    points_pixel = points.clone()
    # Denormalize X (dim 0): [-1, 1] -> pixel
    points_pixel[:, 0, :] = points[:, 0, :] * (w - 1) / 2 + 1 + (w - 1) / 2
    # Denormalize Y (dim 1): [-1, 1] -> pixel
    points_pixel[:, 1, :] = points[:, 1, :] * (h - 1) / 2 + 1 + (h - 1) / 2

    return points_pixel
