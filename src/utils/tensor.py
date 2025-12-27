"""Tensor utilities for batch processing."""
from __future__ import annotations

import torch


class BatchToDevice:
    """Move tensors in a batch dictionary to the specified device.

    Args:
        use_cuda: Whether to use CUDA if available.
    """

    def __init__(self, use_cuda: bool = True):
        self.device = torch.device(
            'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        )

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
