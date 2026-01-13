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


class Muon(Optimizer):
    """Muon optimizer - Momentum Orthogonalized by Newton-schulz.

    A momentum-based optimizer that applies Newton-Schulz orthogonalization
    to the momentum buffer for improved training dynamics.

    Reference: https://github.com/KellerJordan/Muon

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default: 0.02).
        momentum: Momentum factor (default: 0.95).
        nesterov: Whether to use Nesterov momentum (default: True).
        ns_steps: Number of Newton-Schulz iteration steps (default: 5).
        adamw_params: Parameters to use AdamW instead of Muon (default: None).
        adamw_lr: Learning rate for AdamW parameters (default: 3e-4).
        adamw_betas: Betas for AdamW (default: (0.95, 0.95)).
        adamw_eps: Epsilon for AdamW (default: 1e-8).
        adamw_wd: Weight decay for AdamW (default: 0).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params=None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )
        params_list = list(params)
        super().__init__(params_list, defaults)

        # AdamW for non-matrix parameters (biases, norms, embeddings, etc.)
        self.adamw_params = adamw_params or []
        if self.adamw_params:
            self.adamw = torch.optim.AdamW(
                self.adamw_params,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_wd,
            )
        else:
            self.adamw = None

    @staticmethod
    def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """Apply Newton-Schulz iteration to orthogonalize matrix G.

        Computes the orthogonal matrix closest to G in Frobenius norm.
        """
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)

        # Reshape to 2D if needed
        orig_shape = G.shape
        if G.ndim > 2:
            G = G.reshape(G.shape[0], -1)

        # Transpose if needed to have more rows than columns
        transpose = G.shape[0] < G.shape[1]
        if transpose:
            G = G.T

        # Normalize
        G_norm = G.norm() + eps
        G = G / G_norm

        # Newton-Schulz iterations
        for _ in range(steps):
            A = G @ G.T
            B = b * A + c * (A @ A)
            G = a * G + B @ G

        # Restore original shape
        if transpose:
            G = G.T
        G = G.reshape(orig_shape)

        return G * G_norm

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad

                # Initialize momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                # Apply Newton-Schulz orthogonalization for 2D+ tensors
                if g.ndim >= 2 and g.shape[0] > 1 and g.shape[1] > 1:
                    update = self._newton_schulz(buf, steps=ns_steps)
                else:
                    update = buf

                if nesterov:
                    update = update.add(g, alpha=momentum)

                p.add_(update, alpha=-lr)

        # Step AdamW for non-matrix parameters
        if self.adamw is not None:
            self.adamw.step()

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients."""
        super().zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)


def train_epoch(
    epoch: int,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    pair_transform: Callable,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
    use_amp: bool = True,
    profile: bool = False,
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
        use_amp: Whether to use automatic mixed precision (BF16).
        profile: Whether to enable profiling to measure bottlenecks.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    device = device if device is not None else get_device()

    # Setup AMP context
    amp_enabled = use_amp and device.type == 'cuda'
    autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if amp_enabled else torch.autocast(device_type='cuda', enabled=False)

    total_loss = 0.0
    start_time = time()

    # Profiling setup
    if profile:
        timings = {'data': [], 'forward': [], 'backward': [], 'optim': []}
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    batch_to_device = BatchToDevice(device=device)

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

    if profile and device.type == 'cuda':
        torch.cuda.synchronize()
    t_iter_start = time()

    for batch in pbar:
        # Profiling: data loading time
        if profile:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_data = time()
            timings['data'].append(t_data - t_iter_start)

        optimizer.zero_grad()

        # Transform and move to device
        batch = pair_transform(batch)
        batch = batch_to_device(batch)

        # Forward pass with mixed precision
        with autocast_ctx:
            theta_AB, theta_BA, theta_AC, theta_CA = model(batch)
            loss = loss_fn(theta_AB, theta_BA, theta_AC, theta_CA, batch['theta_GT'])

        if profile:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_forward = time()
            timings['forward'].append(t_forward - t_data)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if profile:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_backward = time()
            timings['backward'].append(t_backward - t_forward)

        optimizer.step()

        if profile:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_optim = time()
            timings['optim'].append(t_optim - t_backward)
            t_iter_start = t_optim

        total_loss += loss.item()

        if show_progress:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    elapsed = time() - start_time

    if show_progress:
        tqdm.write(f'  Train Loss: {avg_loss:.4f} ({elapsed:.1f}s)')

    # Print profiling results
    if profile:
        _print_profiling_results(timings, device)

    return avg_loss


def _print_profiling_results(timings: dict, device: torch.device) -> None:
    """Print profiling results summary."""
    print("\n" + "=" * 50)
    print("PROFILING RESULTS")
    print("=" * 50)

    # Calculate averages (skip first iteration as warmup)
    for name, times in timings.items():
        if len(times) > 1:
            times_no_warmup = times[1:]
            avg_ms = sum(times_no_warmup) / len(times_no_warmup) * 1000
            print(f"  {name:12}: {avg_ms:8.2f} ms avg")

    # Total time per iteration
    total_times = []
    for i in range(1, min(len(t) for t in timings.values())):
        total = sum(timings[k][i] for k in timings)
        total_times.append(total)
    if total_times:
        avg_total = sum(total_times) / len(total_times) * 1000
        print(f"  {'TOTAL':12}: {avg_total:8.2f} ms avg")
        print(f"  {'Throughput':12}: {1000 / avg_total:8.2f} iter/s")

    # GPU memory usage
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  GPU Memory Peak: {peak_memory:.2f} GB")

    print("=" * 50 + "\n")


def validate(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    pair_transform: Callable,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
    use_amp: bool = True,
) -> float:
    """Run validation on the dataset.

    Args:
        model: The model to evaluate.
        loss_fn: Loss function module.
        dataloader: Validation data loader.
        pair_transform: Transform function for generating validation pairs.
        device: Device to use. If None, auto-detects (CUDA > MPS > CPU).
        show_progress: Whether to show progress bar.
        use_amp: Whether to use automatic mixed precision (BF16).

    Returns:
        Average validation loss.
    """
    model.eval()
    device = device if device is not None else get_device()

    # Setup AMP context
    amp_enabled = use_amp and device.type == 'cuda'
    autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if amp_enabled else torch.autocast(device_type='cuda', enabled=False)

    total_loss = 0.0

    batch_to_device = BatchToDevice(device=device)

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
            # Transform and move to device
            batch = pair_transform(batch)
            batch = batch_to_device(batch)

            # Forward pass with mixed precision
            with autocast_ctx:
                theta_AB, theta_BA, theta_AC, theta_CA = model(batch)
                loss = loss_fn(theta_AB, theta_BA, theta_AC, theta_CA, batch['theta_GT'])

            total_loss += loss.item()

            if show_progress:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)

    if show_progress:
        tqdm.write(f'  Val Loss:   {avg_loss:.4f}')

    return avg_loss
