"""Checkpoint save/load utilities."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class _PlaceholderConfig:
    """Placeholder dataclass for TrainConfig and similar classes."""
    pass


def save_checkpoint(state: dict, file_path: str | Path) -> None:
    """Save model checkpoint to file.

    Args:
        state: Dictionary containing model state_dict and other info.
        file_path: Path to save checkpoint.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, file_path)


def _remap_checkpoint_keys(state_dict: dict) -> dict:
    """Remap checkpoint keys from old naming convention to new.

    Handles the following remappings:
    - FeatureExtraction -> feature_extraction
    - FeatureRegression -> regression
    - FeatureCorrelation -> correlation
    """
    key_mapping = {
        'FeatureExtraction.': 'feature_extraction.',
        'FeatureRegression.': 'regression.',
        'FeatureCorrelation.': 'correlation.',
    }

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in key_mapping.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        new_state_dict[new_key] = value

    return new_state_dict


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = 'cpu',
    strict: bool = True,
) -> dict:
    """Load checkpoint weights into model.

    Args:
        model: The model instance to load weights into.
        checkpoint_path: Path to checkpoint file.
        device: Device to map checkpoint to.
        strict: If True, strictly enforce that the keys in state_dict match.

    Returns:
        The checkpoint dict (for accessing epoch, optimizer state, etc.).
    """
    # Inject placeholder class for TrainConfig to handle checkpoints saved with it
    main_module = sys.modules.get('__main__')
    if main_module and not hasattr(main_module, 'TrainConfig'):
        setattr(main_module, 'TrainConfig', _PlaceholderConfig)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remap keys from old naming convention
    state_dict = _remap_checkpoint_keys(state_dict)

    # Load state dict
    model.load_state_dict(state_dict, strict=strict)

    # Move model to target device
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)

    print(f"Loaded checkpoint from {checkpoint_path}")
    return checkpoint
