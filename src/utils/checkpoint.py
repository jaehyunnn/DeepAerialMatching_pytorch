"""Checkpoint save/load utilities."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(state: dict, file_path: str | Path) -> None:
    """Save model checkpoint to file.

    Args:
        state: Dictionary containing model state_dict and other info.
        file_path: Path to save checkpoint.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, file_path)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = 'cpu',
) -> dict:
    """Load checkpoint weights into model.

    Args:
        model: The model instance to load weights into.
        checkpoint_path: Path to checkpoint file.
        device: Device to map checkpoint to.

    Returns:
        The checkpoint dict (for accessing epoch, optimizer state, etc.).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint
