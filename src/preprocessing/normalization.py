"""Image normalization utilities."""
from __future__ import annotations

import torch

# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizeImageDict:
    """Normalize tensor images in a dictionary.

    Args:
        image_keys: Dictionary keys of images to normalize.
        normalize_range: If True, divide image by 255.0 first.
    """

    def __init__(self, image_keys: list[str], normalize_range: bool = True):
        self.image_keys = image_keys
        self.normalize_range = normalize_range

    def __call__(self, sample: dict) -> dict:
        for key in self.image_keys:
            image = sample[key]
            if self.normalize_range:
                image = image / 255.0
            sample[key] = normalize_image(image)
        return sample


def normalize_image(
    image: torch.Tensor,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """Normalize image tensor with ImageNet statistics.

    Args:
        image: Input tensor of shape (C, H, W) or (B, C, H, W).
        mean: Channel means for normalization.
        std: Channel standard deviations for normalization.

    Returns:
        Normalized image tensor.
    """
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device)

    if image.dim() == 4:  # (B, C, H, W)
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    else:  # (C, H, W)
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)

    return (image - mean_t) / std_t


def denormalize_image(
    image: torch.Tensor,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """Denormalize image tensor (inverse of normalize_image).

    Args:
        image: Normalized tensor of shape (C, H, W) or (B, C, H, W).
        mean: Channel means used for normalization.
        std: Channel standard deviations used for normalization.

    Returns:
        Denormalized image tensor.
    """
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device)

    if image.dim() == 4:  # (B, C, H, W)
        mean_t = mean_t.view(1, -1, 1, 1)
        std_t = std_t.view(1, -1, 1, 1)
    else:  # (C, H, W)
        mean_t = mean_t.view(-1, 1, 1)
        std_t = std_t.view(-1, 1, 1)

    return image * std_t + mean_t
