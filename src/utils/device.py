"""Device detection and management utilities."""
from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device: The best available device for computation.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_device_name() -> str:
    """Get human-readable device name.

    Returns:
        Device name string (e.g., 'CUDA', 'MPS (Apple Silicon)', 'CPU').
    """
    if torch.cuda.is_available():
        return f'CUDA ({torch.cuda.get_device_name(0)})'
    elif torch.backends.mps.is_available():
        return 'MPS (Apple Silicon)'
    return 'CPU'


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return torch.backends.mps.is_available()


def is_accelerated() -> bool:
    """Check if any GPU acceleration is available (CUDA or MPS)."""
    return torch.cuda.is_available() or torch.backends.mps.is_available()
