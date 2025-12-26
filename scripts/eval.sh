#!/bin/bash
# Deep Aerial Matching - Evaluation Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Default values
MODEL_PATH="checkpoints/checkpoint_seresnext101.pt"
CNN_TYPE="se_resnext101"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --cnn)
            CNN_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./eval.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH    Path to model checkpoint (default: checkpoints/checkpoint_seresnext101.pt)"
            echo "  --cnn TYPE      CNN type: se_resnext101, resnext101, densenet169, resnet101 (default: se_resnext101)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Please run install.sh first to download pretrained models.${NC}"
    exit 1
fi

echo -e "${GREEN}Running evaluation...${NC}"
echo "  Model: $MODEL_PATH"
echo "  CNN:   $CNN_TYPE"
echo ""

python src/cli/eval_pck.py --model-aff "$MODEL_PATH" --feature-extraction-cnn "$CNN_TYPE"
