#!/bin/bash
# =============================================================================
# Deep Aerial Matching - Training Script
# =============================================================================
#
# Usage:
#   ./scripts/train.sh                    # Use default settings below
#   ./scripts/train.sh --help             # Show all available options
#   ./scripts/train.sh --num-epochs 50    # Override specific options
#
# =============================================================================

# -----------------------------------------------------------------------------
# Default Configuration (modify these values as needed)
# -----------------------------------------------------------------------------

# Dataset
DATASET="datasets/training_data"

# Model
BACKBONE="dinov3"          # Options: resnet101, resnext101, se_resnext101, densenet169, dinov3
GEOMETRIC_MODEL="affine"
FREEZE_BACKBONE=true             # true: freeze backbone, false: train backbone
CORRELATION_TYPE="dot"            # Options: dot, cross_attention (LoFTR-style)

# Training
NUM_EPOCHS=100
BATCH_SIZE=8
LEARNING_RATE=0.0004
WEIGHT_DECAY=0.0
SEED=1
NUM_WORKERS=4

# Loss
USE_MSE_LOSS=false                # true: MSE loss, false: Grid loss

# Checkpoint
CHECKPOINT_DIR="checkpoints"
RESUME=""                         # Path to checkpoint to resume (leave empty to start fresh)

# -----------------------------------------------------------------------------
# Script Logic (no need to modify below)
# -----------------------------------------------------------------------------

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Show help if requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo -e "${GREEN}Deep Aerial Matching - Training${NC}"
    echo ""
    echo "Usage: ./scripts/train.sh [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Default settings (edit script to change):${NC}"
    echo "  Dataset:      $DATASET"
    echo "  Backbone:     $BACKBONE"
    echo "  Epochs:       $NUM_EPOCHS"
    echo "  Batch size:   $BATCH_SIZE"
    echo "  LR:           $LEARNING_RATE"
    echo ""
    echo -e "${YELLOW}Available options:${NC}"
    echo "  --training-dataset PATH   Path to training dataset"
    echo "  --backbone MODEL          resnet101|resnext101|se_resnext101|densenet169|dinov3"
    echo "  --num-epochs N            Number of training epochs"
    echo "  --batch-size N            Training batch size"
    echo "  --lr FLOAT                Learning rate"
    echo "  --weight-decay FLOAT      Weight decay"
    echo "  --seed N                  Random seed"
    echo "  --num-workers N           Data loading workers"
    echo "  --use-mse-loss            Use MSE loss instead of grid loss"
    echo "  --freeze-backbone         Freeze backbone weights during training"
    echo "  --correlation-type TYPE   dot|cross_attention (LoFTR-style)"
    echo "  --resume PATH             Resume from checkpoint"
    echo "  --checkpoint-dir PATH     Directory to save checkpoints"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./scripts/train.sh"
    echo "  ./scripts/train.sh --backbone resnet101 --num-epochs 50"
    echo "  ./scripts/train.sh --resume checkpoints/best_se_resnext101.pt"
    echo ""
    exit 0
fi

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Parse command line arguments to override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --training-dataset) DATASET="$2"; shift 2 ;;
        --backbone) BACKBONE="$2"; shift 2 ;;
        --geometric-model) GEOMETRIC_MODEL="$2"; shift 2 ;;
        --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --use-mse-loss) USE_MSE_LOSS=true; shift ;;
        --freeze-backbone) FREEZE_BACKBONE=true; shift ;;
        --correlation-type) CORRELATION_TYPE="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Build arguments from defaults
ARGS=""
ARGS="$ARGS --training-dataset $DATASET"
ARGS="$ARGS --backbone $BACKBONE"
ARGS="$ARGS --geometric-model $GEOMETRIC_MODEL"
ARGS="$ARGS --num-epochs $NUM_EPOCHS"
ARGS="$ARGS --batch-size $BATCH_SIZE"
ARGS="$ARGS --lr $LEARNING_RATE"
ARGS="$ARGS --weight-decay $WEIGHT_DECAY"
ARGS="$ARGS --seed $SEED"
ARGS="$ARGS --num-workers $NUM_WORKERS"
ARGS="$ARGS --checkpoint-dir $CHECKPOINT_DIR"
ARGS="$ARGS --correlation-type $CORRELATION_TYPE"

if [ "$USE_MSE_LOSS" = true ]; then
    ARGS="$ARGS --use-mse-loss"
fi

if [ "$FREEZE_BACKBONE" = true ]; then
    ARGS="$ARGS --freeze-backbone"
fi

if [ -n "$RESUME" ]; then
    ARGS="$ARGS --resume $RESUME"
fi

# Print configuration
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deep Aerial Matching - Training${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Dataset:      $DATASET"
echo "  Backbone:     $BACKBONE"
echo "  Correlation:  $CORRELATION_TYPE"
echo "  Freeze:       $FREEZE_BACKBONE"
echo "  Epochs:       $NUM_EPOCHS"
echo "  Batch size:   $BATCH_SIZE"
echo "  LR:           $LEARNING_RATE"
echo "  Checkpoints:  $CHECKPOINT_DIR"
if [ -n "$RESUME" ]; then
    echo "  Resume from:  $RESUME"
fi
echo ""

# Run training
python src/cli/train.py $ARGS
