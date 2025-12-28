"""Tensor utilities for batch processing."""
from __future__ import annotations

from typing import Optional

import torch

from .device import get_device


class BatchToDevice:
    """Move tensors in a batch dictionary to the specified device.

    Automatically detects the best available device (CUDA > MPS > CPU).

    Args:
        device: Optional explicit device. If None, auto-detects best device.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else get_device()

    def __call__(self, batch: dict) -> dict:
        """Move all tensors in batch to device.

        Args:
            batch: Dictionary containing tensors and other data.

        Returns:
            Dictionary with tensors moved to device.
        """
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
