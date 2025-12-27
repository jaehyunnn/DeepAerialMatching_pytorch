#!/bin/bash
# =============================================================================
# Deep Aerial Matching - Evaluation Script (PCK)
# =============================================================================
#
# Usage:
#   ./scripts/eval.sh                           # Use default settings below
#   ./scripts/eval.sh --help                    # Show all available options
#   ./scripts/eval.sh --backbone resnet101      # Override specific options
#
# =============================================================================

# -----------------------------------------------------------------------------
# Default Configuration (modify these values as needed)
# -----------------------------------------------------------------------------

# Model
BACKBONE="se_resnext101"          # Options: resnet101, resnext101, se_resnext101, densenet169, dinov3
MODEL="checkpoints/checkpoint_seresnext101.pt"
CORRELATION_TYPE="dot"            # Options: dot, cross_attention (LoFTR-style)

# Dataset
DATASET_PATH="datasets/evaluation_data"

# Evaluation
BATCH_SIZE=16
NUM_WORKERS=4

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
    echo -e "${GREEN}Deep Aerial Matching - Evaluation (PCK)${NC}"
    echo ""
    echo "Usage: ./scripts/eval.sh [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Default settings (edit script to change):${NC}"
    echo "  Backbone:     $BACKBONE"
    echo "  Model:        $MODEL"
    echo "  Dataset:      $DATASET_PATH"
    echo "  Batch size:   $BATCH_SIZE"
    echo ""
    echo -e "${YELLOW}Available options:${NC}"
    echo "  --backbone MODEL        resnet101|resnext101|se_resnext101|densenet169|dinov3"
    echo "  --model PATH            Path to model checkpoint"
    echo "  --correlation-type TYPE dot|cross_attention (LoFTR-style)"
    echo "  --dataset-path PATH     Path to evaluation dataset"
    echo "  --batch-size N         Evaluation batch size"
    echo "  --num-workers N        Data loading workers"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./scripts/eval.sh"
    echo "  ./scripts/eval.sh --backbone resnet101 --model checkpoints/checkpoint_resnet101.pt"
    echo "  ./scripts/eval.sh --batch-size 32"
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
        --backbone) BACKBONE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --correlation-type) CORRELATION_TYPE="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Check if model file exists
if [ ! -f "$MODEL" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL${NC}"
    echo -e "${YELLOW}Please run install.sh first to download pretrained models.${NC}"
    exit 1
fi

# Build arguments
ARGS=""
ARGS="$ARGS --backbone $BACKBONE"
ARGS="$ARGS --model $MODEL"
ARGS="$ARGS --correlation-type $CORRELATION_TYPE"
ARGS="$ARGS --dataset-path $DATASET_PATH"
ARGS="$ARGS --batch-size $BATCH_SIZE"
ARGS="$ARGS --num-workers $NUM_WORKERS"

# Print configuration
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deep Aerial Matching - Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Backbone:     $BACKBONE"
echo "  Correlation:  $CORRELATION_TYPE"
echo "  Model:        $MODEL"
echo "  Dataset:      $DATASET_PATH"
echo "  Batch size:   $BATCH_SIZE"
echo ""

# Run evaluation
python src/cli/eval.py $ARGS
