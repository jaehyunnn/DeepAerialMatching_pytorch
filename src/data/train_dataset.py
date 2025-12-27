"""Training dataset for aerial image matching."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from transforms import GeometricTnf


@dataclass
class RandomSampleConfig:
    """Configuration for random transformation sampling."""
    translation: float = 0.5
    scale: float = 0.5
    rotation: float = 0.5  # fraction of pi


class TrainDataset(Dataset):
    """Training dataset with synthetic transformation pairs.

    Loads image pairs with ground truth affine transformations for
    training geometric matching networks with strong supervision.

    Args:
        csv_file: Path to CSV file with image names and transformations
        image_path: Directory containing training images
        output_size: Output image size (height, width)
        geometric_model: Type of geometric transformation ('affine')
        transform: Optional transform for post-processing
        random_sample: If True, generate random transformations instead of using CSV
        random_config: Configuration for random transformation sampling
    """

    def __init__(
        self,
        csv_file: str,
        image_path: str,
        output_size: tuple[int, int] = (540, 540),
        geometric_model: str = 'affine',
        transform=None,
        random_sample: bool = False,
        random_config: RandomSampleConfig | None = None,
    ):
        self.out_h, self.out_w = output_size
        self.image_path = image_path
        self.geometric_model = geometric_model
        self.transform = transform
        self.random_sample = random_sample
        self.random_config = random_config or RandomSampleConfig()

        # Load CSV data
        self.data = pd.read_csv(csv_file)
        self.src_img_names = self.data.iloc[:, 0]
        self.trg_img_names = self.data.iloc[:, 1]
        self.theta_array = self.data.iloc[:, 2:].values.astype('float')

        # Image transforms
        self.resize_tnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=1, saturation=1, hue=0.1
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """Get training sample.

        Returns:
            Dictionary containing:
                - src_image: Source image tensor
                - trg_image: Target image tensor
                - trg_image_jit: Color-jittered target image
                - theta: Ground truth transformation (2, 3)
        """
        # Load images
        src_image = self._load_image(self.src_img_names[idx])
        trg_image = self._load_image(self.trg_img_names[idx])
        trg_image_jit = self.color_jitter(trg_image)

        # Get transformation
        theta = self._get_theta(idx)

        # Convert to tensors
        src_tensor = self._image_to_tensor(src_image)
        trg_tensor = self._image_to_tensor(trg_image)
        trg_jit_tensor = self._image_to_tensor(trg_image_jit)
        theta_tensor = torch.Tensor(theta.astype(np.float32))

        # Resize if needed
        src_tensor = self._resize_if_needed(src_tensor)
        trg_tensor = self._resize_if_needed(trg_tensor)
        trg_jit_tensor = self._resize_if_needed(trg_jit_tensor)

        sample = {
            'src_image': src_tensor,
            'trg_image': trg_tensor,
            'trg_image_jit': trg_jit_tensor,
            'theta': theta_tensor,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_image(self, img_name: str) -> Image.Image:
        """Load image from disk."""
        img_path = os.path.join(self.image_path, img_name)
        return Image.open(img_path)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor with 0-255 range."""
        return self.to_tensor(image) * 255

    def _resize_if_needed(self, image: torch.Tensor) -> torch.Tensor:
        """Resize image if dimensions don't match output size."""
        if image.size()[1] != self.out_h or image.size()[2] != self.out_w:
            image = self.resize_tnf(image.unsqueeze(0)).squeeze(0)
        return image

    def _get_theta(self, idx: int) -> np.ndarray:
        """Get transformation matrix for sample.

        Returns theta from CSV or generates random transformation.
        """
        if not self.random_sample:
            theta = self.theta_array[idx, :]
            if self.geometric_model == 'affine':
                theta = theta.reshape(2, 3)
            return theta

        # Generate random affine transformation
        if self.geometric_model == 'affine':
            return self._generate_random_affine()

        raise ValueError(f"Unknown geometric model: {self.geometric_model}")

    def _generate_random_affine(self) -> np.ndarray:
        """Generate random affine transformation matrix.

        Returns:
            Affine transformation matrix of shape (2, 3)
        """
        cfg = self.random_config

        # Random rotation angle
        alpha = (np.random.rand(1) - 0.5) * 2 * np.pi * cfg.rotation

        # Generate random parameters
        theta = np.random.rand(6)

        # Translation (columns 2 and 5)
        theta[[2, 5]] = (theta[[2, 5]] - 0.5) * 2 * cfg.translation

        # Rotation and scale
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        scale_factor = 1 + (theta[:2] - 0.5) * 2 * cfg.scale

        theta[0] = scale_factor[0] * cos_a
        theta[1] = scale_factor[0] * (-sin_a)
        theta[3] = scale_factor[1] * sin_a
        theta[4] = scale_factor[1] * cos_a

        return theta.reshape(2, 3)
