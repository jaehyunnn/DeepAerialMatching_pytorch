"""Training and validation loop utilities."""
from __future__ import annotations

from time import time
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .device import get_device
from .tensor import BatchToDevice


def train_epoch(
    epoch: int,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    pair_transform: Callable,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> float:
    """Run one training epoch.

    Args:
        epoch: Current epoch number.
        model: The model to train.
        loss_fn: Loss function module.
        optimizer: Optimizer instance.
        dataloader: Training data loader.
        pair_transform: Transform function for generating training pairs.
        device: Device to use. If None, auto-detects (CUDA > MPS > CPU).
        show_progress: Whether to show progress bar.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    device = device if device is not None else get_device()
    batch_to_device = BatchToDevice(device=device)

    total_loss = 0.0
    start_time = time()

    # Create progress bar
    if show_progress:
        pbar = tqdm(
            dataloader,
            desc=f'Epoch {epoch:3d} [Train]',
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
    else:
        pbar = dataloader

    for batch in pbar:
        optimizer.zero_grad()

        # Generate training pair and move to device
        batch = pair_transform(batch)
        batch = batch_to_device(batch)

        # Forward pass
        theta_AB, theta_BA, theta_AC, theta_CA = model(batch)
        loss = loss_fn(theta_AB, theta_BA, theta_AC, theta_CA, batch['theta_GT'])

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if show_progress:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    elapsed = time() - start_time

    if show_progress:
        tqdm.write(f'  Train Loss: {avg_loss:.4f} ({elapsed:.1f}s)')

    return avg_loss


def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    pair_transform: Callable,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> float:
    """Run validation on the dataset.

    Args:
        model: The model to evaluate.
        loss_fn: Loss function module.
        dataloader: Validation data loader.
        pair_transform: Transform function for generating validation pairs.
        device: Device to use. If None, auto-detects (CUDA > MPS > CPU).
        show_progress: Whether to show progress bar.

    Returns:
        Average validation loss.
    """
    model.eval()
    device = device if device is not None else get_device()
    batch_to_device = BatchToDevice(device=device)

    total_loss = 0.0

    # Create progress bar
    if show_progress:
        pbar = tqdm(
            dataloader,
            desc='          [Val]  ',
            leave=False,
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
    else:
        pbar = dataloader

    with torch.no_grad():
        for batch in pbar:
            # Generate validation pair and move to device
            batch = pair_transform(batch)
            batch = batch_to_device(batch)

            # Forward pass
            theta_AB, theta_BA, theta_AC, theta_CA = model(batch)
            loss = loss_fn(theta_AB, theta_BA, theta_AC, theta_CA, batch['theta_GT'])

            total_loss += loss.item()

            if show_progress:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)

    if show_progress:
        tqdm.write(f'  Val Loss:   {avg_loss:.4f}')

    return avg_loss
