#!/bin/bash
# =============================================================================
# Deep Aerial Matching - Demo Script
# =============================================================================
#
# Usage:
#   ./scripts/demo.sh                           # Use default settings below
#   ./scripts/demo.sh --help                    # Show all available options
#   ./scripts/demo.sh --backbone resnet101      # Override specific options
#
# =============================================================================

# -----------------------------------------------------------------------------
# Default Configuration (modify these values as needed)
# -----------------------------------------------------------------------------

# Model
BACKBONE="se_resnext101"          # Options: resnet101, resnext101, se_resnext101, densenet169, vit-l/16
MODEL="checkpoints/checkpoint_seresnext101.pt"
VERSION=""                        # Options: v1, v2 (empty = auto-detect based on backbone)

# Images
SOURCE_IMAGE="datasets/demo_img/00_src.jpg"
TARGET_IMAGE="datasets/demo_img/00_tgt.jpg"

# Output
OUTPUT_DIR="results"
NO_DISPLAY=false                  # true: headless mode, false: show plot

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
    echo -e "${GREEN}Deep Aerial Matching - Demo${NC}"
    echo ""
    echo "Usage: ./scripts/demo.sh [OPTIONS]"
    echo ""
    echo -e "${YELLOW}Default settings (edit script to change):${NC}"
    echo "  Backbone:     $BACKBONE"
    echo "  Model:        $MODEL"
    echo "  Source:       $SOURCE_IMAGE"
    echo "  Target:       $TARGET_IMAGE"
    echo "  Output:       $OUTPUT_DIR"
    echo ""
    echo -e "${YELLOW}Available options:${NC}"
    echo "  --backbone MODEL        resnet101|resnext101|se_resnext101|densenet169|vit-l/16"
    echo "  --model PATH            Path to model checkpoint"
    echo "  --version VERSION       v1|v2 (v1: BatchNorm, v2: GroupNorm+dual_softmax, auto-detect if not set)"
    echo "  --source PATH           Path to source image"
    echo "  --target PATH        Path to target image"
    echo "  --output-dir PATH    Directory to save results"
    echo "  --no-display         Do not display results (headless mode)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  ./scripts/demo.sh"
    echo "  ./scripts/demo.sh --backbone resnet101 --model checkpoints/checkpoint_resnet101.pt"
    echo "  ./scripts/demo.sh --source my_src.jpg --target my_tgt.jpg"
    echo "  ./scripts/demo.sh --no-display"
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
        --version) VERSION="$2"; shift 2 ;;
        --source) SOURCE_IMAGE="$2"; shift 2 ;;
        --target) TARGET_IMAGE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --no-display) NO_DISPLAY=true; shift ;;
        *) shift ;;
    esac
done

# Check if model checkpoint exists, if not, offer to download
if [ ! -f "$MODEL" ]; then
    echo -e "${YELLOW}Model checkpoint not found: $MODEL${NC}"
    echo -e "${YELLOW}Would you like to download it? [y/N]${NC}"
    read -p "> " download_choice

    if [[ "$download_choice" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Downloading $BACKBONE checkpoint...${NC}"
        python -c "from src.data.download import download_model; download_model('$BACKBONE')" || {
            echo -e "${RED}Failed to download model${NC}"
            exit 1
        }
    else
        echo -e "${RED}Cannot proceed without model checkpoint${NC}"
        exit 1
    fi
fi

# Build arguments
ARGS=""
ARGS="$ARGS --backbone $BACKBONE"
ARGS="$ARGS --model $MODEL"
if [ -n "$VERSION" ]; then
    ARGS="$ARGS --version $VERSION"
fi
ARGS="$ARGS --source $SOURCE_IMAGE"
ARGS="$ARGS --target $TARGET_IMAGE"
ARGS="$ARGS --output-dir $OUTPUT_DIR"

if [ "$NO_DISPLAY" = true ]; then
    ARGS="$ARGS --no-display"
fi

# Print configuration
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Deep Aerial Matching - Demo${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Backbone:     $BACKBONE"
echo "  Version:      ${VERSION:-auto}"
echo "  Model:        $MODEL"
echo "  Source:       $SOURCE_IMAGE"
echo "  Target:       $TARGET_IMAGE"
echo "  Output:       $OUTPUT_DIR"
echo "  No display:   $NO_DISPLAY"
echo ""

# Run demo
python src/cli/demo.py $ARGS
