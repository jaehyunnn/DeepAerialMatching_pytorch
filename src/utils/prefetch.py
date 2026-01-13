"""GPU data prefetching utilities for faster training."""
from __future__ import annotations

from typing import Callable, Iterator, Optional

import torch
from torch.utils.data import DataLoader


class CUDAPrefetcher:
    """Prefetches batches to GPU asynchronously for faster training.

    Uses CUDA streams to overlap data transfer with model computation.
    While the model processes batch N, batch N+1 is being transferred to GPU.

    Args:
        loader: The DataLoader to prefetch from.
        device: Target CUDA device.
        transform: Optional transform to apply on GPU after transfer.

    Example:
        >>> prefetcher = CUDAPrefetcher(train_loader, device, pair_transform)
        >>> for batch in prefetcher:
        ...     output = model(batch)
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        transform: Optional[Callable] = None,
    ):
        self.loader = loader
        self.device = device
        self.transform = transform
        self.stream = torch.cuda.Stream(device=device)
        self._iterator: Optional[Iterator] = None
        self._next_batch: Optional[dict] = None

    def __iter__(self) -> 'CUDAPrefetcher':
        self._iterator = iter(self.loader)
        self._preload()
        return self

    def __next__(self) -> dict:
        # Wait for the prefetch stream to complete
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

        batch = self._next_batch
        if batch is None:
            raise StopIteration

        # Ensure tensors are safe to use on current stream
        for v in batch.values():
            if torch.is_tensor(v):
                v.record_stream(torch.cuda.current_stream(self.device))

        # Start loading next batch
        self._preload()
        return batch

    def __len__(self) -> int:
        return len(self.loader)

    def _preload(self) -> None:
        """Load next batch asynchronously on prefetch stream."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next_batch = None
            return

        with torch.cuda.stream(self.stream):
            # Transfer to GPU with non_blocking
            self._next_batch = {}
            for k, v in batch.items():
                if torch.is_tensor(v):
                    self._next_batch[k] = v.to(self.device, non_blocking=True)
                else:
                    self._next_batch[k] = v

            # Apply transform on GPU if provided
            if self.transform is not None:
                self._next_batch = self.transform(self._next_batch)
