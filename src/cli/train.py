"""Deep Aerial Matching training script."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from os.path import basename

# Add src to path for direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import AerialNetTwoStream, TransformedGridLoss
from data import TrainDataset, download_train
from transforms import SynthPairTnf
from preprocessing import NormalizeImageDict
from utils import (
    train_epoch, validate, save_checkpoint, str_to_bool,
    get_device, get_device_name,
)


@dataclass
class TrainConfig:
    """Training configuration."""
    # Paths
    dataset_path: str = 'datasets/training_data'
    checkpoint_dir: str = 'checkpoints'

    # Model
    backbone: str = 'se_resnext101'
    geometric_model: str = 'affine'
    freeze_backbone: bool = False
    correlation_type: str = 'dot'

    # Training
    num_epochs: int = 100
    batch_size: int = 12
    lr: float = 0.0004
    weight_decay: float = 0.0
    seed: int = 1
    num_workers: int = 4

    # Loss
    use_mse_loss: bool = False

    # Data augmentation
    random_sample: bool = False

    # Resume
    resume_from: str | None = None

    @property
    def dataset_name(self) -> str:
        return basename(self.dataset_path.rstrip('/'))

    @property
    def loss_type(self) -> str:
        return 'mse_loss' if self.use_mse_loss else 'grid_loss'


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    transform = NormalizeImageDict(['src_image', 'trg_image', 'trg_image_jit'])

    train_dataset = TrainDataset(
        csv_file=f'{config.dataset_path}/train_pair.csv',
        image_path=f'{config.dataset_path}/',
        geometric_model=config.geometric_model,
        transform=transform,
        random_sample=config.random_sample,
    )

    val_dataset = TrainDataset(
        csv_file=f'{config.dataset_path}/val_pair.csv',
        image_path=f'{config.dataset_path}/',
        geometric_model=config.geometric_model,
        transform=transform,
        random_sample=config.random_sample,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader


def create_model(config: TrainConfig, device: torch.device) -> AerialNetTwoStream:
    """Create and optionally load model."""
    model = AerialNetTwoStream(
        freeze_backbone=config.freeze_backbone,
        geometric_model=config.geometric_model,
        backbone=config.backbone,
        correlation_type=config.correlation_type,
        device=device,
    )

    if config.resume_from and os.path.exists(config.resume_from):
        checkpoint = torch.load(config.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    return model


def create_loss(config: TrainConfig, device: torch.device) -> nn.Module:
    """Create loss function."""
    if config.use_mse_loss:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = TransformedGridLoss(geometric_model=config.geometric_model)
    return loss_fn.to(device)


def run_training(config: TrainConfig):
    """Run the training loop."""
    # Auto-detect device (CUDA > MPS > CPU)
    device = get_device()
    print(f'Using device: {get_device_name()}')

    # Set seed
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.seed)

    # Download dataset if needed
    if not os.path.exists(config.dataset_path):
        download_train('datasets')

    # Create components
    model = create_model(config, device)
    loss_fn = create_loss(config, device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    pair_tnf = SynthPairTnf(geometric_model=config.geometric_model, device=device)

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    print(f'\nTraining for {config.num_epochs} epochs...\n')

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_epoch(
            epoch=epoch,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader=train_loader,
            pair_transform=pair_tnf,
            device=device,
        )
        val_loss = validate(
            model=model,
            loss_fn=loss_fn,
            dataloader=val_loader,
            pair_transform=pair_tnf,
            device=device,
        )

        # Save checkpoint
        checkpoint_name = (
            f"{config.checkpoint_dir}/"
            f"{config.geometric_model}_{config.loss_type}_{config.backbone}_"
            f"{config.dataset_name}_epoch_{epoch}.pt"
        )
        save_checkpoint({
            'epoch': epoch,
            'config': config,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_name)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f"{config.checkpoint_dir}/best_{config.backbone}.pt"
            save_checkpoint({
                'epoch': epoch,
                'config': config,
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f'  ★ New best model saved (val_loss: {val_loss:.4f})')

        print()  # Blank line between epochs

    print(f'Training complete! Best val_loss: {best_val_loss:.4f}')


def parse_args() -> TrainConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Deep Aerial Matching Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Paths
    parser.add_argument('--training-dataset', type=str, default='datasets/training_data',
                        help='Path to training dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')

    # Model
    parser.add_argument('--backbone', type=str, default='se_resnext101',
                        choices=['resnet101', 'resnext101', 'se_resnext101', 'densenet169', 'dinov3'],
                        help='Feature extraction backbone')
    parser.add_argument('--geometric-model', type=str, default='affine',
                        help='Geometric transformation model')
    parser.add_argument('--freeze-backbone', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Freeze backbone weights during training')
    parser.add_argument('--correlation-type', type=str, default='dot',
                        choices=['dot', 'cross_attention'],
                        help='Correlation type (dot: simple dot product, cross_attention: LoFTR-style)')

    # Training
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0004,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Loss
    parser.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Use MSE loss instead of grid loss')

    # Data augmentation
    parser.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Use random transformation sampling')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    args = parser.parse_args()

    return TrainConfig(
        dataset_path=args.training_dataset,
        checkpoint_dir=args.checkpoint_dir,
        backbone=args.backbone,
        geometric_model=args.geometric_model,
        freeze_backbone=args.freeze_backbone,
        correlation_type=args.correlation_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        use_mse_loss=args.use_mse_loss,
        random_sample=args.random_sample,
        resume_from=args.resume,
    )


def main():
    """Main entry point."""
    config = parse_args()
    run_training(config)


if __name__ == '__main__':
    main()
