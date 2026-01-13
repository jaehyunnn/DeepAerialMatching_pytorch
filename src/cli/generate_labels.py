"""Random Transformation Label Generator.

Generates random affine or homography transformation parameters for training data.
Refactored from: https://github.com/jaehyunnn/RndTnfGen
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# Add src to path for direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class TransformConfig:
    """Configuration for transformation generation."""

    num_samples: int = 10000
    transform_type: Literal["affine", "homography"] = "affine"
    # Affine parameters
    max_translation: float = 0.5
    max_scale: float = 0.5
    max_rotation: float = 0.5  # Fraction of pi (0.5 = 180 degrees)
    # Homography parameters
    max_perspective: float = 0.1
    # Output
    output_path: str = "outputs/theta.csv"
    add_rotation: bool = False
    input_csv: str = ""  # For add_rotation mode


class TransformGenerator:
    """Generates random geometric transformation parameters."""

    def __init__(self, config: TransformConfig):
        self.config = config

    def generate(self) -> np.ndarray:
        """Generate transformation parameters based on config."""
        if self.config.transform_type == "affine":
            return self._generate_affine()
        else:
            return self._generate_homography()

    def _generate_affine(self) -> np.ndarray:
        """Generate random affine transformation parameters.

        Affine matrix format: [a11, a12, tx, a21, a22, ty]
        Where the transformation is:
            | a11  a12  tx |
            | a21  a22  ty |

        Returns:
            np.ndarray: Shape (N, 6) affine parameters
        """
        n = self.config.num_samples
        t = self.config.max_translation
        s = self.config.max_scale
        r = self.config.max_rotation

        # Random rotation angles
        alpha = (np.random.rand(n) - 0.5) * 2 * np.pi * r

        # Random scale factors
        scale = 1 + (np.random.rand(n, 2) - 0.5) * 2 * s

        # Random translations
        translation = (np.random.rand(n, 2) - 0.5) * 2 * t

        # Build affine parameters
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)

        theta = np.zeros((n, 6), dtype=np.float32)
        theta[:, 0] = scale[:, 0] * cos_a  # a11
        theta[:, 1] = scale[:, 0] * (-sin_a)  # a12
        theta[:, 2] = translation[:, 0]  # tx
        theta[:, 3] = scale[:, 1] * sin_a  # a21
        theta[:, 4] = scale[:, 1] * cos_a  # a22
        theta[:, 5] = translation[:, 1]  # ty

        return theta

    def _generate_homography(self) -> np.ndarray:
        """Generate random homography transformation parameters.

        Homography is computed from 4-point correspondences with random perturbations.
        Output format: [h11, h12, tx, h21, h22, ty, h31, h32] (h33 = 1)

        Returns:
            np.ndarray: Shape (N, 8) homography parameters
        """
        n = self.config.num_samples
        p = self.config.max_perspective

        # Base 4 corners: (0,0), (1,0), (0,1), (1,1)
        base_points = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32)

        theta_list = []
        for _ in range(n):
            # Add random perturbation to corners
            perturbed = base_points + (np.random.rand(8) - 0.5) * 2 * p
            h = self._points_to_homography(base_points, perturbed)
            theta_list.append(h)

        return np.array(theta_list, dtype=np.float32)

    @staticmethod
    def _points_to_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Compute homography from 4-point correspondences using DLT.

        Args:
            src: Source points [x1, x2, x3, x4, y1, y2, y3, y4]
            dst: Destination points [x1', x2', x3', x4', y1', y2', y3', y4']

        Returns:
            np.ndarray: Homography parameters [h11, h12, h13, h21, h22, h23, h31, h32]
        """
        P = []
        for i in range(4):
            x, y = src[i], src[i + 4]
            x_p, y_p = dst[i], dst[i + 4]
            P.append([x, y, 1, 0, 0, 0, -x * x_p, -y * x_p])
            P.append([0, 0, 0, x, y, 1, -x * y_p, -y * y_p])

        P = np.array(P, dtype=np.float32)
        b = dst.astype(np.float32)

        # Solve P @ h = b
        h = np.linalg.lstsq(P, b, rcond=None)[0]
        return h

    def add_rotation_to_existing(self, theta: np.ndarray) -> np.ndarray:
        """Add random rotation to existing transformation parameters.

        Args:
            theta: Existing transformation parameters (N, 6) for affine or (N, 8) for homography

        Returns:
            np.ndarray: Rotated transformation parameters
        """
        n = len(theta)
        r = self.config.max_rotation

        # Generate random rotations
        alpha = np.random.rand(n) * 2 * np.pi * r
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)

        # Build rotation matrices
        rot = np.zeros((n, 3, 3), dtype=np.float32)
        rot[:, 0, 0] = cos_a
        rot[:, 0, 1] = -sin_a
        rot[:, 1, 0] = sin_a
        rot[:, 1, 1] = cos_a
        rot[:, 2, 2] = 1.0

        # Reshape theta to 3x3 matrices
        if theta.shape[1] == 6:  # Affine
            mat = np.zeros((n, 3, 3), dtype=np.float32)
            mat[:, 0, :2] = theta[:, :2].reshape(n, 2)
            mat[:, 0, 2] = theta[:, 2]
            mat[:, 1, :2] = theta[:, 3:5].reshape(n, 2)
            mat[:, 1, 2] = theta[:, 5]
            mat[:, 2, 2] = 1.0
        else:  # Homography
            mat = np.concatenate([theta, np.ones((n, 1))], axis=1).reshape(n, 3, 3)

        # Apply rotation
        result = np.matmul(rot, mat)

        if theta.shape[1] == 6:
            return result[:, :2, :].reshape(n, 6)
        else:
            return result[:, :, :].reshape(n, 9)[:, :8]

    def save(self, theta: np.ndarray, path: str | Path) -> None:
        """Save transformation parameters to CSV.

        Args:
            theta: Transformation parameters
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if theta.shape[1] == 6:
            header = "a11,a12,tx,a21,a22,ty"
        else:
            header = "h11,h12,tx,h21,h22,ty,h31,h32"

        np.savetxt(path, theta, delimiter=",", header=header, fmt="%.6f", comments="")
        print(f"Saved {len(theta)} transformations to {path}")

    def load(self, path: str | Path) -> np.ndarray:
        """Load transformation parameters from CSV.

        Args:
            path: Input file path

        Returns:
            np.ndarray: Transformation parameters
        """
        return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)


class DataPairExtractor:
    """Extract image pair information from dataset directories."""

    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)

    def extract(
        self, extensions: tuple[str, ...] = (".jpg", ".png", ".tif")
    ) -> list[tuple[str, str]]:
        """Extract matching image pairs from source and target directories.

        Args:
            extensions: Image file extensions to include

        Returns:
            List of (source_path, target_path) tuples
        """
        pairs = []

        for ext in extensions:
            source_files = sorted(self.source_dir.glob(f"*{ext}"))
            for src_file in source_files:
                tgt_file = self.target_dir / src_file.name
                if tgt_file.exists():
                    pairs.append((str(src_file), str(tgt_file)))

        return pairs

    def save_pairs(self, pairs: list[tuple[str, str]], output_path: str | Path) -> None:
        """Save image pairs to CSV file.

        Args:
            pairs: List of (source, target) path tuples
            output_path: Output CSV file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = np.array(pairs)
        np.savetxt(output_path, data, delimiter=",", header="source,target", fmt="%s", comments="")
        print(f"Saved {len(pairs)} image pairs to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate random transformation labels for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate random transformations")
    gen_parser.add_argument(
        "-n", "--num-samples", type=int, default=10000, help="Number of samples to generate"
    )
    gen_parser.add_argument(
        "-t",
        "--type",
        choices=["affine", "homography"],
        default="affine",
        help="Transformation type",
    )
    gen_parser.add_argument(
        "--max-translation", type=float, default=0.5, help="Maximum translation ratio"
    )
    gen_parser.add_argument("--max-scale", type=float, default=0.5, help="Maximum scale ratio")
    gen_parser.add_argument(
        "--max-rotation", type=float, default=0.5, help="Maximum rotation (fraction of pi)"
    )
    gen_parser.add_argument(
        "--max-perspective",
        type=float,
        default=0.1,
        help="Maximum perspective distortion (homography only)",
    )
    gen_parser.add_argument(
        "-o", "--output", type=str, default="outputs/theta.csv", help="Output file path"
    )

    # Add rotation command
    rot_parser = subparsers.add_parser(
        "add-rotation", help="Add rotation to existing transformations"
    )
    rot_parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input CSV file with transformations"
    )
    rot_parser.add_argument(
        "--max-rotation", type=float, default=0.5, help="Maximum rotation (fraction of pi)"
    )
    rot_parser.add_argument(
        "-o", "--output", type=str, default="outputs/rotated_theta.csv", help="Output file path"
    )

    # Extract pairs command
    pair_parser = subparsers.add_parser(
        "extract-pairs", help="Extract image pairs from directories"
    )
    pair_parser.add_argument(
        "-s", "--source-dir", type=str, required=True, help="Source image directory"
    )
    pair_parser.add_argument(
        "-t", "--target-dir", type=str, required=True, help="Target image directory"
    )
    pair_parser.add_argument(
        "-o", "--output", type=str, default="outputs/pairs.csv", help="Output file path"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.command == "generate":
        config = TransformConfig(
            num_samples=args.num_samples,
            transform_type=args.type,
            max_translation=args.max_translation,
            max_scale=args.max_scale,
            max_rotation=args.max_rotation,
            max_perspective=args.max_perspective,
        )
        generator = TransformGenerator(config)
        theta = generator.generate()
        generator.save(theta, args.output)

        # Print statistics
        print("\nStatistics:")
        print(f"  Shape: {theta.shape}")
        print(f"  Mean:  {theta.mean(axis=0)}")
        print(f"  Std:   {theta.std(axis=0)}")

    elif args.command == "add-rotation":
        config = TransformConfig(max_rotation=args.max_rotation)
        generator = TransformGenerator(config)
        theta = generator.load(args.input)
        rotated = generator.add_rotation_to_existing(theta)
        generator.save(rotated, args.output)

    elif args.command == "extract-pairs":
        extractor = DataPairExtractor(args.source_dir, args.target_dir)
        pairs = extractor.extract()
        extractor.save_pairs(pairs, args.output)

    else:
        print("Please specify a command: generate, add-rotation, or extract-pairs")
        print("Use --help for more information")


if __name__ == "__main__":
    main()
