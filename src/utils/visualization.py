"""Visualization utilities for image comparison."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def create_checkerboard(
    image1: NDArray[np.floating],
    image2: NDArray[np.floating],
    grid_size: int = 7,
) -> NDArray[np.float32]:
    """Create a checkerboard visualization from two images.

    Alternates between tiles from image1 and image2 in a checkerboard pattern.
    Useful for visualizing alignment between a source and warped image.

    Args:
        image1: First image (H, W, C), values in [0, 1].
        image2: Second image (H, W, C), values in [0, 1] or [0, 255].
        grid_size: Number of tiles per row/column.

    Returns:
        Checkerboard image (H', W', C) with float32 values in [0, 1].

    Raises:
        ValueError: If images have different shapes.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Images must have the same shape, got {image1.shape} and {image2.shape}"
        )

    # Normalize uint8 to float
    if image2.dtype == np.uint8:
        image2 = image2.astype(np.float32) / 255.0

    height, width, channels = image1.shape
    tile_h = height // grid_size
    tile_w = width // grid_size

    # Output size is aligned to tile boundaries
    out_h = tile_h * grid_size
    out_w = tile_w * grid_size
    output = np.zeros((out_h, out_w, channels), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            h_start = tile_h * i
            h_end = h_start + tile_h
            w_start = tile_w * j
            w_end = w_start + tile_w

            # Alternate between images based on checkerboard pattern
            source = image1 if (i + j) % 2 == 0 else image2
            output[h_start:h_end, w_start:w_end, :] = source[h_start:h_end, w_start:w_end, :]

    return output
