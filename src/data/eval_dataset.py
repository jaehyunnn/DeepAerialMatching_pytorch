"""PCK evaluation datasets for aerial image matching."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset

from transforms import GeometricTnf


class BasePCKDataset(Dataset, ABC):
    """Base class for PCK evaluation datasets.

    Provides common functionality for loading image pairs and keypoint
    coordinates for Percentage of Correct Keypoints (PCK) evaluation.
    """

    def __init__(
        self,
        csv_file: str,
        image_path: str,
        output_size: tuple[int, int] = (240, 240),
        transform=None,
    ):
        """Initialize dataset.

        Args:
            csv_file: Path to CSV file containing image pairs and annotations
            image_path: Base path to image directory
            output_size: Output image size (height, width)
            transform: Optional transform to apply to samples
        """
        self.out_h, self.out_w = output_size
        self.image_path = image_path
        self.transform = transform

        # Load CSV data
        self.data = pd.read_csv(csv_file)
        self.src_names = self.data.iloc[:, 0]
        self.trg_names = self.data.iloc[:, 1]
        self.src_point_coords = self.data.iloc[:, 2:42].values.astype('float')

        # Geometric transform for resizing (use CPU in dataloader workers)
        self.resize_tnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, device=torch.device('cpu'))

    def __len__(self) -> int:
        return len(self.data)

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Get sample at index."""
        pass

    def _load_image(self, img_names: pd.Series, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess an image.

        Args:
            img_names: Series of image filenames
            idx: Index of image to load

        Returns:
            Tuple of (image tensor, original image size)
        """
        img_path = os.path.join(self.image_path, img_names[idx])
        image = io.imread(img_path)

        # Store original size
        im_size = torch.Tensor(np.asarray(image.shape).astype(np.float32))

        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))

        # Resize to output size
        image = self.resize_tnf(image).squeeze(0)

        return image, im_size

    def _load_points(self, point_coords: np.ndarray, idx: int) -> torch.Tensor:
        """Load keypoint coordinates.

        Args:
            point_coords: Array of point coordinates
            idx: Index of sample

        Returns:
            Point coordinates tensor of shape (2, 20)
        """
        points = point_coords[idx, :].reshape(2, 20)
        return torch.Tensor(points.astype(np.float32))


class PCKEvalDataset(BasePCKDataset):
    """PCK evaluation dataset with ground truth transformation matrix.

    CSV format: src_name, trg_name, src_points (40 values), theta_GT (6 values)

    Used for evaluating image matching by comparing predicted transformation
    against ground truth affine transformation matrix.
    """

    def __init__(
        self,
        csv_file: str,
        image_path: str,
        output_size: tuple[int, int] = (540, 540),
        transform=None,
    ):
        super().__init__(csv_file, image_path, output_size, transform)
        self.theta_GT = self.data.iloc[:, 42:].values.astype('float')

    def __getitem__(self, idx: int) -> dict:
        """Get sample with ground truth transformation.

        Returns:
            Dictionary containing:
                - source_image: Source image tensor
                - target_image: Target image tensor
                - source_im_size: Original source image size
                - target_im_size: Original target image size
                - source_points: Source keypoint coordinates (2, 20)
                - theta_GT: Ground truth affine transformation (2, 3)
                - L_pck: PCK normalization factor
        """
        src_image, src_im_size = self._load_image(self.src_names, idx)
        trg_image, trg_im_size = self._load_image(self.trg_names, idx)

        src_points = self._load_points(self.src_point_coords, idx)
        theta_GT = self._load_theta(idx)

        # L_pck: normalization factor (max dimension of source image)
        L_pck = torch.FloatTensor([torch.max(src_im_size)])

        sample = {
            'source_image': src_image,
            'target_image': trg_image,
            'source_im_size': src_im_size,
            'target_im_size': trg_im_size,
            'source_points': src_points,
            'theta_GT': theta_GT,
            'L_pck': L_pck,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _load_theta(self, idx: int) -> torch.Tensor:
        """Load ground truth transformation matrix.

        Args:
            idx: Index of sample

        Returns:
            Affine transformation matrix of shape (2, 3)
        """
        theta = self.theta_GT[idx, :].reshape(2, 3)
        return torch.Tensor(theta.astype(np.float32))


class PCKEvalDatasetV2(BasePCKDataset):
    """PCK evaluation dataset with target point coordinates.

    CSV format: src_name, trg_name, src_points (40 values), trg_points (40 values)

    Used for direct keypoint correspondence evaluation without
    transformation matrix.
    """

    def __init__(
        self,
        csv_file: str,
        image_path: str,
        output_size: tuple[int, int] = (240, 240),
        transform=None,
    ):
        super().__init__(csv_file, image_path, output_size, transform)
        self.trg_point_coords = self.data.iloc[:, 42:].values.astype('float')

    def __getitem__(self, idx: int) -> dict:
        """Get sample with target point coordinates.

        Returns:
            Dictionary containing:
                - source_image: Source image tensor
                - target_image: Target image tensor
                - source_im_size: Original source image size
                - target_im_size: Original target image size
                - source_points: Source keypoint coordinates (2, 20)
                - target_points: Target keypoint coordinates (2, 20)
        """
        src_image, src_im_size = self._load_image(self.src_names, idx)
        trg_image, trg_im_size = self._load_image(self.trg_names, idx)

        src_points = self._load_points(self.src_point_coords, idx)
        trg_points = self._load_points(self.trg_point_coords, idx)

        sample = {
            'source_image': src_image,
            'target_image': trg_image,
            'source_im_size': src_im_size,
            'target_im_size': trg_im_size,
            'source_points': src_points,
            'target_points': trg_points,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
