"""Deep Aerial Matching training script."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from os.path import basename

# Add src to path for direct script execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from models import AerialNetTwoStream, TransformedGridLoss
from data import TrainDataset, download_train
from transforms import SynthPairTnf
from preprocessing import NormalizeImageDict
from utils import (
    train_epoch, validate, save_checkpoint, str_to_bool,
    get_device, get_device_name,
)
from utils.training import Muon


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
    version: str | None = None  # v1 or v2, auto-detect if None

    # Training
    num_epochs: int = 100
    batch_size: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 1
    num_workers: int = 4
    optimizer: str = 'adamw'  # Options: adamw, muon
    use_amp: bool = True  # Use BF16 mixed precision
    grad_checkpoint: bool = False  # Use gradient checkpointing (ViT only)

    # Loss
    use_mse_loss: bool = False

    # Data augmentation
    random_sample: bool = False

    # Resume
    resume_from: str | None = None

    # Profiling
    profile: bool = False

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

    # Use pin_memory for faster CPU->GPU transfer, persistent_workers to avoid respawning
    use_persistent = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent,
    )

    return train_loader, val_loader


def create_model(config: TrainConfig, device: torch.device) -> tuple[AerialNetTwoStream, int, dict | None, dict | None, str | None]:
    """Create and optionally load model from checkpoint.

    Returns:
        model: The model instance
        start_epoch: Epoch to start training from (1 if fresh, checkpoint_epoch + 1 if resuming)
        optimizer_state: Optimizer state dict if resuming, None otherwise
        scheduler_state: Scheduler state dict if resuming, None otherwise
        wandb_run_id: Wandb run ID if resuming, None otherwise
    """
    model = AerialNetTwoStream(
        freeze_backbone=config.freeze_backbone,
        geometric_model=config.geometric_model,
        backbone=config.backbone,
        version=config.version,
        use_grad_checkpoint=config.grad_checkpoint,
        device=device,
    )

    start_epoch = 1
    optimizer_state = None
    scheduler_state = None
    wandb_run_id = None

    if config.resume_from and os.path.exists(config.resume_from):
        checkpoint = torch.load(config.resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resuming from epoch {checkpoint["epoch"]}, starting at epoch {start_epoch}')

        if 'optimizer' in checkpoint:
            optimizer_state = checkpoint['optimizer']

        if 'scheduler' in checkpoint:
            scheduler_state = checkpoint['scheduler']

        if 'wandb_run_id' in checkpoint:
            wandb_run_id = checkpoint['wandb_run_id']

    return model, start_epoch, optimizer_state, scheduler_state, wandb_run_id


def create_loss(config: TrainConfig, device: torch.device) -> nn.Module:
    """Create loss function."""
    if config.use_mse_loss:
        loss_fn = nn.MSELoss()
    else:
        loss_fn = TransformedGridLoss(geometric_model=config.geometric_model)
    return loss_fn.to(device)


def create_optimizer(config: TrainConfig, model: nn.Module) -> optim.Optimizer:
    """Create optimizer based on config."""
    if config.optimizer == 'muon':
        # Separate matrix params (for Muon) and non-matrix params (for AdamW)
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Use AdamW for biases, norms, and 1D parameters
            if param.ndim < 2 or 'bias' in name or 'norm' in name or 'bn' in name:
                adamw_params.append(param)
            else:
                muon_params.append(param)

        optimizer = Muon(
            muon_params,
            lr=config.lr,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_params=adamw_params,
            adamw_lr=config.lr * 0.1,  # Lower LR for non-matrix params
            adamw_wd=config.weight_decay,
        )
    else:  # adamw (default)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    return optimizer


def print_model_parameters(model: nn.Module) -> tuple[int, int]:
    """Print total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f'\n{"="*50}')
    print(f'Model Parameters:')
    print(f'  Total:     {total_params:,}')
    print(f'  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)')
    print(f'  Frozen:    {frozen_params:,} ({100*frozen_params/total_params:.1f}%)')
    print(f'{"="*50}\n')

    return total_params, trainable_params


def save_config_md(
    config: TrainConfig,
    checkpoint_dir: str,
    device_name: str,
    amp_enabled: bool,
    total_params: int,
    trainable_params: int,
    train_samples: int,
    val_samples: int,
    img_size: tuple[int, int],
) -> None:
    """Save training configuration as markdown file."""
    md_content = f"""# Training Configuration

## Session Info
- **Timestamp**: {os.path.basename(checkpoint_dir)}
- **Device**: {device_name}
- **Mixed Precision (BF16)**: {'Enabled' if amp_enabled else 'Disabled'}

## Model
| Parameter | Value |
|-----------|-------|
| Backbone | {config.backbone} |
| Version | {config.version or 'auto'} |
| Geometric Model | {config.geometric_model} |
| Freeze Backbone | {config.freeze_backbone} |
| Total Parameters | {total_params:,} |
| Trainable Parameters | {trainable_params:,} |

## Training
| Parameter | Value |
|-----------|-------|
| Optimizer | {config.optimizer} |
| Learning Rate | {config.lr} |
| Weight Decay | {config.weight_decay} |
| Batch Size | {config.batch_size} |
| Epochs | {config.num_epochs} |
| Seed | {config.seed} |
| Num Workers | {config.num_workers} |

## Dataset
| Parameter | Value |
|-----------|-------|
| Path | {config.dataset_path} |
| Train Samples | {train_samples:,} |
| Val Samples | {val_samples:,} |
| Image Size | {img_size[0]} x {img_size[1]} |

## Loss
| Parameter | Value |
|-----------|-------|
| Loss Type | {config.loss_type} |
| Use MSE Loss | {config.use_mse_loss} |

## Data Augmentation
| Parameter | Value |
|-----------|-------|
| Random Sample | {config.random_sample} |

## Resume
| Parameter | Value |
|-----------|-------|
| Resume From | {config.resume_from or 'None'} |
"""

    config_path = os.path.join(checkpoint_dir, 'config.md')
    with open(config_path, 'w') as f:
        f.write(md_content)
    print(f'Config saved to: {config_path}')


def run_training(config: TrainConfig):
    """Run the training loop."""
    # Auto-detect device (CUDA > MPS > CPU)
    device = get_device()
    amp_enabled = config.use_amp and device.type == 'cuda'
    print(f'Using device: {get_device_name()}')
    print(f'Mixed precision (BF16): {"Enabled" if amp_enabled else "Disabled"}')

    # Set seed
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(config.seed)

    # Download dataset if needed
    if not os.path.exists(config.dataset_path):
        download_train('datasets')

    # Create components (load checkpoint first to get wandb_run_id)
    model, start_epoch, optimizer_state, scheduler_state, wandb_run_id = create_model(config, device)

    # Initialize wandb (resume if we have a run_id from checkpoint)
    if wandb_run_id:
        wandb.init(
            project='DeepAerialMatching',
            id=wandb_run_id,
            resume='must',
            config=asdict(config),
        )
        print(f'Resuming wandb run: {wandb_run_id}')
    else:
        wandb.init(
            project='DeepAerialMatching',
            config=asdict(config),
            name=f'{config.backbone}_{config.geometric_model}_{config.loss_type}',
        )
    total_params, trainable_params = print_model_parameters(model)
    loss_fn = create_loss(config, device)
    optimizer = create_optimizer(config, model)

    # Load optimizer state if resuming
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        print('Optimizer state restored from checkpoint')

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    # Load scheduler state if resuming
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        print('Scheduler state restored from checkpoint')

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    pair_tnf = SynthPairTnf(geometric_model=config.geometric_model, device=device)

    # Print dataset info
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    img_size = (train_dataset.out_h, train_dataset.out_w)
    print(f'\nDataset:')
    print(f'  Train samples: {len(train_dataset):,}')
    print(f'  Val samples:   {len(val_dataset):,}')
    print(f'  Image size:    {img_size[0]} x {img_size[1]}')

    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    checkpoint_dir = os.path.join(config.checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f'Checkpoints will be saved to: {checkpoint_dir}')

    # Save config as markdown
    save_config_md(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device_name=get_device_name(),
        amp_enabled=amp_enabled,
        total_params=total_params,
        trainable_params=trainable_params,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        img_size=img_size,
    )

    # Training loop
    best_val_loss = float('inf')

    if start_epoch > 1:
        print(f'\nResuming training from epoch {start_epoch} to {config.num_epochs}...\n')
    else:
        print(f'\nTraining for {config.num_epochs} epochs...\n')

    for epoch in range(start_epoch, config.num_epochs + 1):
        train_loss = train_epoch(
            epoch=epoch,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader=train_loader,
            pair_transform=pair_tnf,
            device=device,
            use_amp=config.use_amp,
            profile=config.profile,
        )
        val_loss = validate(
            model=model,
            loss_fn=loss_fn,
            dataloader=val_loader,
            pair_transform=pair_tnf,
            device=device,
            use_amp=config.use_amp,
        )

        # Step scheduler
        scheduler.step()

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0],
        })

        # Save checkpoint
        checkpoint_name = (
            f"{checkpoint_dir}/"
            f"{config.geometric_model}_{config.loss_type}_{config.backbone}_"
            f"{config.dataset_name}_epoch_{epoch}.pt"
        )
        save_checkpoint({
            'epoch': epoch,
            'config': config,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'wandb_run_id': wandb.run.id,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_name)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = f"{checkpoint_dir}/best_{config.backbone}.pt"
            save_checkpoint({
                'epoch': epoch,
                'config': config,
                'state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f'  ★ New best model saved (val_loss: {val_loss:.4f})')

        print()  # Blank line between epochs

    print(f'Training complete! Best val_loss: {best_val_loss:.4f}')
    wandb.finish()


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
                        choices=['resnet101', 'resnext101', 'se_resnext101', 'densenet169', 'vit-l/16'],
                        help='Feature extraction backbone')
    parser.add_argument('--geometric-model', type=str, default='affine',
                        help='Geometric transformation model')
    parser.add_argument('--freeze-backbone', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Freeze backbone weights during training')
    parser.add_argument('--version', type=str, default=None,
                        choices=['v1', 'v2'],
                        help='Model version (v1: BatchNorm, v2: GroupNorm+dual_softmax, auto-detect if not set)')

    # Training
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.0004,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'muon'],
                        help='Optimizer (adamw: AdamW, muon: Muon with Newton-Schulz)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use-amp', type=str_to_bool, nargs='?', const=True, default=True,
                        help='Use BF16 mixed precision training (CUDA only)')
    parser.add_argument('--grad-checkpoint', action='store_true',
                        help='Enable gradient checkpointing for memory efficiency (ViT backbones only)')

    # Loss
    parser.add_argument('--use-mse-loss', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Use MSE loss instead of grid loss')

    # Data augmentation
    parser.add_argument('--random-sample', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Use random transformation sampling')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Profiling
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling to measure training bottlenecks (runs 1 epoch only)')

    args = parser.parse_args()

    return TrainConfig(
        dataset_path=args.training_dataset,
        checkpoint_dir=args.checkpoint_dir,
        backbone=args.backbone,
        geometric_model=args.geometric_model,
        freeze_backbone=args.freeze_backbone,
        version=args.version,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        seed=args.seed,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        grad_checkpoint=args.grad_checkpoint,
        use_mse_loss=args.use_mse_loss,
        random_sample=args.random_sample,
        resume_from=args.resume,
        profile=args.profile,
    )


def main():
    """Main entry point."""
    config = parse_args()
    run_training(config)


if __name__ == '__main__':
    main()
