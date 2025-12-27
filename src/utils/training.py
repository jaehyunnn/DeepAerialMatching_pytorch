"""Training and validation loop utilities."""
from __future__ import annotations

from time import time
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .tensor import BatchToDevice


def train_epoch(
    epoch: int,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    pair_transform: Callable,
    use_cuda: bool = True,
    log_interval: int = 50,
) -> float:
    """Run one training epoch.

    Args:
        epoch: Current epoch number.
        model: The model to train.
        loss_fn: Loss function module.
        optimizer: Optimizer instance.
        dataloader: Training data loader.
        pair_transform: Transform function for generating training pairs.
        use_cuda: Whether to use CUDA.
        log_interval: Batches between log messages.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    batch_to_device = BatchToDevice(use_cuda=use_cuda)

    total_loss = 0.0
    start_time = time()

    for batch_idx, batch in enumerate(dataloader):
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

        if batch_idx % log_interval == 0:
            progress = 100.0 * batch_idx / len(dataloader)
            print(
                f'Train Epoch: {epoch} [{batch_idx}/{len(dataloader)} ({progress:.0f}%)]\t'
                f'Loss: {loss.item():.6f}'
            )

    avg_loss = total_loss / len(dataloader)
    elapsed = time() - start_time
    print(f'Train set: Average loss: {avg_loss:.4f} --- {elapsed:.2f}s')

    return avg_loss


def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    pair_transform: Callable,
    use_cuda: bool = True,
) -> float:
    """Run validation on the dataset.

    Args:
        model: The model to evaluate.
        loss_fn: Loss function module.
        dataloader: Validation data loader.
        pair_transform: Transform function for generating validation pairs.
        use_cuda: Whether to use CUDA.

    Returns:
        Average validation loss.
    """
    model.eval()
    batch_to_device = BatchToDevice(use_cuda=use_cuda)

    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # Generate validation pair and move to device
            batch = pair_transform(batch)
            batch = batch_to_device(batch)

            # Forward pass
            theta_AB, theta_BA, theta_AC, theta_CA = model(batch)
            loss = loss_fn(theta_AB, theta_BA, theta_AC, theta_CA, batch['theta_GT'])

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Validation set: Average loss: {avg_loss:.4f}')

    return avg_loss
