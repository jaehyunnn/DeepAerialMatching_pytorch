"""Deep Aerial Matching demo script."""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass

# Add src to path for direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from torchvision.transforms import Normalize

from models import AerialNetSingleStream
from preprocessing import normalize_image
from transforms import GeometricTnf, theta2homogeneous
from utils import load_checkpoint, create_checkerboard, get_device, get_device_name

warnings.filterwarnings('ignore')


@dataclass
class DemoConfig:
    """Demo configuration."""
    # Model
    backbone: str = 'se_resnext101'
    model_path: str = 'checkpoints/checkpoint_seresnext101.pt'
    version: str | None = None  # v1 or v2, auto-detect if None

    # Images
    source_image: str = 'datasets/demo_img/00_src.jpg'
    target_image: str = 'datasets/demo_img/00_tgt.jpg'

    # Output
    output_dir: str = 'results'
    show_plot: bool = True


def create_model(config: DemoConfig, device: torch.device) -> AerialNetSingleStream:
    """Create and load model."""
    print('Creating model...')
    model = AerialNetSingleStream(
        device=device,
        geometric_model='affine',
        backbone=config.backbone,
        version=config.version,
    )

    print(f'Loading trained model weights from {config.model_path}...')
    load_checkpoint(model, config.model_path, device=device)

    return model


def load_and_preprocess_image(image_path: str, resize_tnf: GeometricTnf, device: torch.device) -> tuple:
    """Load and preprocess an image for inference."""
    image = io.imread(image_path)

    # Convert to torch Tensor
    image_tensor = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image_tensor = torch.Tensor(image_tensor.astype(np.float32) / 255.0)

    # Resize and normalize
    image_tensor = resize_tnf(image_tensor)
    image_tensor = normalize_image(image_tensor)

    return image, image_tensor.to(device)


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy image to tensor."""
    image_tensor = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image_tensor = torch.Tensor(image_tensor.astype(np.float32) / 255.0)
    return image_tensor.to(device)


def compute_ensemble_theta(theta: torch.Tensor, theta_inv: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Compute ensemble of forward and inverse transformations."""
    batch_size = theta.size(0)
    theta_inv = theta_inv.view(-1, 2, 3)

    homogeneous_row = torch.tensor(
        [[0, 0, 1]], dtype=theta_inv.dtype, device=device
    ).expand(batch_size, 1, 3)

    theta_inv = torch.cat((theta_inv, homogeneous_row), dim=1)
    theta_2 = torch.linalg.inv(theta_inv)[:, :2, :].reshape(-1, 6)

    return (theta + theta_2) / 2


def run_inference(model: AerialNetSingleStream, batch: dict, device: torch.device) -> tuple:
    """Run two-stage affine inference."""
    model.eval()

    # First affine transformation
    theta_aff, theta_aff_inv = model(batch)
    theta_aff_ensemble = compute_ensemble_theta(theta_aff, theta_aff_inv, device)

    return theta_aff_ensemble


def run_two_stage_inference(
    model: AerialNetSingleStream,
    source_image: np.ndarray,
    batch: dict,
    resize_tnf: GeometricTnf,
    aff_tnf: GeometricTnf,
    device: torch.device
) -> tuple:
    """Run two-stage affine inference with refinement."""
    model.eval()

    # First affine transformation
    theta_aff, theta_aff_inv = model(batch)
    theta_aff_ensemble = compute_ensemble_theta(theta_aff, theta_aff_inv, device)

    # Warp source image with first transformation
    warped_image = aff_tnf(image_to_tensor(source_image, device), theta_aff_ensemble.view(-1, 2, 3))

    # Second affine transformation (refinement)
    source_image_2 = normalize_image(resize_tnf(warped_image.cpu())).to(device)
    theta_aff_aff, theta_aff_aff_inv = model({
        'source_image': source_image_2,
        'target_image': batch['target_image']
    })
    theta_aff_aff_ensemble = compute_ensemble_theta(theta_aff_aff, theta_aff_aff_inv, device)

    # Combine transformations
    theta_aff_homo = theta2homogeneous(theta_aff_ensemble)
    theta_aff_aff_homo = theta2homogeneous(theta_aff_aff_ensemble)
    theta_combined = torch.bmm(theta_aff_aff_homo, theta_aff_homo).view(-1, 9)[:, :6]

    return theta_aff_ensemble, theta_combined, warped_image


def save_results(
    config: DemoConfig,
    source_image: np.ndarray,
    target_image: np.ndarray,
    theta_aff: torch.Tensor,
    theta_combined: torch.Tensor,
    aff_tnf: GeometricTnf,
    device: torch.device
) -> tuple:
    """Save result images."""
    os.makedirs(config.output_dir, exist_ok=True)

    target_float = np.float32(target_image / 255.0)

    # First affine result
    warped_aff = aff_tnf(image_to_tensor(source_image, device), theta_aff.view(-1, 2, 3))
    result_aff = warped_aff.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    result_aff = np.clip(result_aff, 0, 1)
    io.imsave(f'{config.output_dir}/aff.jpg', (result_aff * 255).astype(np.uint8))

    # Two-stage affine result
    warped_aff_aff = aff_tnf(image_to_tensor(source_image, device), theta_combined.view(-1, 2, 3))
    result_aff_aff = warped_aff_aff.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    result_aff_aff = np.clip(result_aff_aff, 0, 1)
    io.imsave(f'{config.output_dir}/aff_aff.jpg', (result_aff_aff * 255).astype(np.uint8))

    # Overlay (alpha + beta = 1.2, so clip result)
    overlay = cv2.addWeighted(src1=result_aff, alpha=0.4, src2=target_float, beta=0.8, gamma=0)
    overlay = np.clip(overlay, 0, 1)
    io.imsave(f'{config.output_dir}/aff_overlay.jpg', (overlay * 255).astype(np.uint8))

    # Checkboard
    checkboard = create_checkerboard(result_aff, target_float)
    io.imsave(f'{config.output_dir}/aff_checkboard.jpg', (checkboard * 255).astype(np.uint8))

    return result_aff, result_aff_aff, overlay, checkboard, target_float


def display_results(
    source_image: np.ndarray,
    target_image: np.ndarray,
    result_aff: np.ndarray,
    result_aff_aff: np.ndarray,
    checkboard: np.ndarray,
    overlay: np.ndarray
):
    """Display results in a matplotlib figure."""
    fig, axs = plt.subplots(2, 3)

    axs[0][0].imshow(source_image)
    axs[0][0].set_title('Source')
    axs[0][1].imshow(target_image)
    axs[0][1].set_title('Target')
    axs[0][2].imshow(result_aff)
    axs[0][2].set_title('Affine')

    axs[1][0].imshow(result_aff_aff)
    axs[1][0].set_title('Affine X 2')
    axs[1][1].imshow(checkboard)
    axs[1][1].set_title('Affine Checkboard')
    axs[1][2].imshow(overlay)
    axs[1][2].set_title('Affine Overlay')

    for i in range(2):
        for j in range(3):
            axs[i][j].axis('off')

    fig.set_dpi(300)
    plt.show()


def run_demo(config: DemoConfig):
    """Run the demo pipeline."""
    # Auto-detect device (CUDA > MPS > CPU)
    device = get_device()
    print(f'Using device: {get_device_name()}')

    # Create model
    model = create_model(config, device)

    # Create transforms (resize on CPU for compatibility)
    resize_tnf = GeometricTnf(out_h=240, out_w=240, device=torch.device('cpu'))

    # Load images
    source_image, source_tensor = load_and_preprocess_image(
        config.source_image, resize_tnf, device
    )
    target_image, target_tensor = load_and_preprocess_image(
        config.target_image, resize_tnf, device
    )

    # Create affine transform for output size
    aff_tnf = GeometricTnf(
        geometric_model='affine',
        out_h=target_image.shape[0],
        out_w=target_image.shape[1],
        device=device,
    )

    # Prepare batch
    batch = {'source_image': source_tensor, 'target_image': target_tensor}

    # Run inference
    start_time = time.time()
    theta_aff, theta_combined, _ = run_two_stage_inference(
        model, source_image, batch, resize_tnf, aff_tnf, device
    )
    elapsed = time.time() - start_time

    print(f'\nExecution time: {elapsed:.4f} seconds')

    # Save results
    result_aff, result_aff_aff, overlay, checkboard, target_float = save_results(
        config, source_image, target_image, theta_aff, theta_combined, aff_tnf, device
    )

    print(f'Results saved to {config.output_dir}/')

    # Display results
    if config.show_plot:
        display_results(source_image, target_float, result_aff, result_aff_aff, checkboard, overlay)

    print('Done!')


def parse_args() -> DemoConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Deep Aerial Matching Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument('--backbone', type=str, default='se_resnext101',
                        choices=['resnet101', 'resnext101', 'se_resnext101', 'densenet169', 'vit-l/16'],
                        help='Feature extraction backbone')
    parser.add_argument('--model', type=str, default='checkpoints/checkpoint_seresnext101.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--version', type=str, default=None,
                        choices=['v1', 'v2'],
                        help='Model version (v1: BatchNorm, v2: GroupNorm+dual_softmax, auto-detect if not set)')

    # Images
    parser.add_argument('--source', type=str, default='datasets/demo_img/00_src.jpg',
                        help='Path to source image')
    parser.add_argument('--target', type=str, default='datasets/demo_img/00_tgt.jpg',
                        help='Path to target image')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display results (headless mode)')

    args = parser.parse_args()

    return DemoConfig(
        backbone=args.backbone,
        model_path=args.model,
        version=args.version,
        source_image=args.source,
        target_image=args.target,
        output_dir=args.output_dir,
        show_plot=not args.no_display,
    )


def main():
    """Main entry point."""
    config = parse_args()
    run_demo(config)


if __name__ == '__main__':
    main()
