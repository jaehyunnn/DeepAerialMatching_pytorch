"""Deep Aerial Matching evaluation script (PCK metrics)."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

# Add src to path for direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import AerialNetSingleStream
from data import PCKEvalDataset, download_eval
from preprocessing import NormalizeImageDict
from utils import BatchToDevice, load_checkpoint, get_device, get_device_name
from transforms import GeometricTnf, SynthPairTnfPCK, PointTnf, points_to_unit_coords, points_to_pixel_coords


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Model
    backbone: str = 'se_resnext101'
    model_path: str = 'checkpoints/checkpoint_seresnext101.pt'
    version: str | None = None  # v1 or v2, auto-detect if None

    # Dataset
    dataset_path: str = 'datasets/evaluation_data'

    # Evaluation
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class PCKResults:
    """PCK evaluation results."""
    pck_aff_001: float = 0.0
    pck_aff_003: float = 0.0
    pck_aff_005: float = 0.0
    pck_aff_aff_001: float = 0.0
    pck_aff_aff_003: float = 0.0
    pck_aff_aff_005: float = 0.0


def create_model(config: EvalConfig, device: torch.device) -> AerialNetSingleStream:
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


def create_dataloader(config: EvalConfig, device: torch.device) -> DataLoader:
    """Create evaluation dataloader."""
    # Download dataset if needed
    download_eval('datasets')

    dataset = PCKEvalDataset(
        csv_file=os.path.join(config.dataset_path, 'test_pairs.csv'),
        image_path=config.dataset_path,
        transform=NormalizeImageDict(['source_image', 'target_image'])
    )

    # Use larger batch size for GPU acceleration
    batch_size = config.batch_size if device.type != 'cpu' else 1

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return dataloader, len(dataset)


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


def correct_keypoints(
    source_points: torch.Tensor,
    warped_points: torch.Tensor,
    L_pck: torch.Tensor,
    tau: float = 0.01
) -> tuple[int, int]:
    """Compute correct keypoints based on PCK threshold."""
    point_distance = torch.pow(
        torch.sum(torch.pow(source_points - warped_points, 2), 1), 0.5
    ).squeeze(1)

    L_pck_mat = L_pck.to(point_distance.device).expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * tau)

    num_correct = torch.sum(correct_points)
    num_total = correct_points.numel()

    return num_correct, num_total


def evaluate_batch(
    model: AerialNetSingleStream,
    batch: dict,
    pair_tnf: SynthPairTnfPCK,
    pt: PointTnf,
    aff_tnf: GeometricTnf,
    device: torch.device
) -> dict:
    """Evaluate a single batch and return keypoint results."""
    source_im_size = batch['source_im_size']
    theta_GT = batch['theta_GT']
    source_points = batch['source_points']

    # Apply pair transformation
    batch_tnf = pair_tnf({
        'src_image': batch['source_image'],
        'trg_image': batch['target_image'],
        'theta': theta_GT
    })
    batch['source_image'] = batch_tnf['source_image']
    batch['target_image'] = batch_tnf['target_image']

    # Normalize source points
    source_points_norm = points_to_unit_coords(source_points, source_im_size / 2)

    model.eval()

    # First affine transformation
    theta_aff, theta_aff_inv = model(batch)
    theta_aff_ensemble = compute_ensemble_theta(theta_aff, theta_aff_inv, device)

    # Warp points with GT and estimated transformations
    warped_points_GT_norm = pt.affine_transform(theta_GT, source_points_norm)
    warped_points_aff_norm = pt.affine_transform(theta_aff_ensemble, source_points_norm)

    warped_points_GT = points_to_pixel_coords(warped_points_GT_norm, source_im_size / 2)
    warped_points_aff = points_to_pixel_coords(warped_points_aff_norm, source_im_size / 2)

    # Second affine transformation (refinement)
    warped_image_aff = aff_tnf(batch['source_image'], theta_aff_ensemble.view(-1, 2, 3))
    theta_aff_aff, theta_aff_aff_inv = model({
        'source_image': warped_image_aff,
        'target_image': batch['target_image']
    })
    theta_aff_aff_ensemble = compute_ensemble_theta(theta_aff_aff, theta_aff_aff_inv, device)

    # Warp points with two-stage transformation
    warped_points_aff_aff_norm = pt.affine_transform(theta_aff_aff_ensemble, source_points_norm)
    warped_points_aff_aff_norm = pt.affine_transform(theta_aff_ensemble, warped_points_aff_aff_norm)
    warped_points_aff_aff = points_to_pixel_coords(warped_points_aff_aff_norm, source_im_size / 2)

    return {
        'warped_GT': warped_points_GT.data,
        'warped_aff': warped_points_aff.data,
        'warped_aff_aff': warped_points_aff_aff.data,
        'L_pck': batch['L_pck'].data,
    }


def run_evaluation(config: EvalConfig) -> PCKResults:
    """Run the evaluation pipeline."""
    # Auto-detect device (CUDA > MPS > CPU)
    device = get_device()
    print(f'Using device: {get_device_name()}')

    # Create model
    model = create_model(config, device)

    # Create dataloader
    dataloader, dataset_size = create_dataloader(config, device)

    # Create transforms
    pair_tnf = SynthPairTnfPCK(geometric_model='affine', device=device)
    batch_to_device = BatchToDevice(device=device)
    pt = PointTnf()
    aff_tnf = GeometricTnf(geometric_model='affine', device=device)

    print(f'\nEvaluating on {dataset_size} pairs...')

    # Accumulators for each tau threshold
    stats = {
        'aff': {0.01: [0, 0], 0.03: [0, 0], 0.05: [0, 0]},
        'aff_aff': {0.01: [0, 0], 0.03: [0, 0], 0.05: [0, 0]},
    }

    for batch in tqdm(dataloader):
        batch = batch_to_device(batch)

        results = evaluate_batch(model, batch, pair_tnf, pt, aff_tnf, device)

        # Compute PCK for each threshold
        for tau in [0.01, 0.03, 0.05]:
            # Single affine
            correct, total = correct_keypoints(
                results['warped_GT'], results['warped_aff'], results['L_pck'], tau
            )
            stats['aff'][tau][0] += correct.item()
            stats['aff'][tau][1] += total

            # Two-stage affine
            correct, total = correct_keypoints(
                results['warped_GT'], results['warped_aff_aff'], results['L_pck'], tau
            )
            stats['aff_aff'][tau][0] += correct.item()
            stats['aff_aff'][tau][1] += total

    # Calculate final PCK values
    results = PCKResults(
        pck_aff_001=stats['aff'][0.01][0] / stats['aff'][0.01][1],
        pck_aff_003=stats['aff'][0.03][0] / stats['aff'][0.03][1],
        pck_aff_005=stats['aff'][0.05][0] / stats['aff'][0.05][1],
        pck_aff_aff_001=stats['aff_aff'][0.01][0] / stats['aff_aff'][0.01][1],
        pck_aff_aff_003=stats['aff_aff'][0.03][0] / stats['aff_aff'][0.03][1],
        pck_aff_aff_005=stats['aff_aff'][0.05][0] / stats['aff_aff'][0.05][1],
    )

    return results


def print_results(results: PCKResults):
    """Print PCK results in a formatted table."""
    print('')
    print('# ================ PCK Results ================ #')
    print('# Tau                   | 0.01  | 0.03  | 0.05  #')
    print('# ----------------------|-------|-------|------ #')
    print(f'# PCK affine            | {results.pck_aff_001:.3f} | {results.pck_aff_003:.3f} | {results.pck_aff_005:.3f} #')
    print(f'# PCK affine (2-times)  | {results.pck_aff_aff_001:.3f} | {results.pck_aff_aff_003:.3f} | {results.pck_aff_aff_005:.3f} #')
    print('# ============================================= #')


def parse_args() -> EvalConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Deep Aerial Matching Evaluation (PCK)',
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

    # Dataset
    parser.add_argument('--dataset-path', type=str, default='datasets/evaluation_data',
                        help='Path to evaluation dataset')

    # Evaluation
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    return EvalConfig(
        backbone=args.backbone,
        model_path=args.model,
        version=args.version,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def main():
    """Main entry point."""
    config = parse_args()
    results = run_evaluation(config)
    print_results(results)
    print('\nDone!')


if __name__ == '__main__':
    main()
