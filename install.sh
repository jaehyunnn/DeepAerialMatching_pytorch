#!/bin/bash
# Deep Aerial Matching - Installation Script
# Uses uv for fast Python environment setup

set -e

echo "=============================================="
echo "  Deep Aerial Matching - Environment Setup"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Hugging Face repo
HF_REPO="jaehyunnn/DeepAerialMatching"

# Check and install uv
install_uv() {
    if command -v uv &> /dev/null; then
        echo -e "${GREEN}uv is already installed${NC}"
        return
    fi

    echo -e "${YELLOW}Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        echo -e "${GREEN}uv installed successfully${NC}"
    else
        echo -e "${RED}Failed to install uv${NC}"
        exit 1
    fi
}

# Install uv if needed
install_uv

echo ""
echo -e "${YELLOW}Setting up Python environment...${NC}"

# Create venv and install dependencies with uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,download]"
uv pip install huggingface_hub

# Create directories
echo ""
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p checkpoints datasets results

# Download pretrained models from Hugging Face
echo ""
echo -e "${YELLOW}Downloading pretrained models from Hugging Face...${NC}"

download_model() {
    local filename=$1
    local model_name=$2
    local output_path="checkpoints/${filename}"

    if [ -f "$output_path" ]; then
        echo -e "${GREEN}$model_name already exists, skipping...${NC}"
        return
    fi

    echo "Downloading $model_name..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='checkpoints/${filename}',
    local_dir='.'
)
"

    if [ -f "$output_path" ]; then
        echo -e "${GREEN}Downloaded $model_name successfully${NC}"
    else
        echo -e "${RED}Failed to download $model_name${NC}"
    fi
}

echo ""
echo "Select models to download:"
echo "  1) ViT-L/16 (Best, v2.0)"
echo "  2) SE-ResNeXt101 (v1)"
echo "  3) ResNeXt101 (v1)"
echo "  4) DenseNet169 (v1)"
echo "  5) ResNet101 (v1)"
echo "  6) All models"
echo "  7) Skip"
echo ""
read -p "Enter choice [1-7] (default: 1): " model_choice
model_choice=${model_choice:-1}

case $model_choice in
    1) download_model "checkpoint_vit-l16.pt" "ViT-L/16" ;;
    2) download_model "checkpoint_seresnext101.pt" "SE-ResNeXt101" ;;
    3) download_model "checkpoint_resnext101.pt" "ResNeXt101" ;;
    4) download_model "checkpoint_densenet169.pt" "DenseNet169" ;;
    5) download_model "checkpoint_resnet101.pt" "ResNet101" ;;
    6)
        download_model "checkpoint_vit-l16.pt" "ViT-L/16"
        download_model "checkpoint_seresnext101.pt" "SE-ResNeXt101"
        download_model "checkpoint_resnext101.pt" "ResNeXt101"
        download_model "checkpoint_densenet169.pt" "DenseNet169"
        download_model "checkpoint_resnet101.pt" "ResNet101"
        ;;
    7) echo -e "${YELLOW}Skipping model download${NC}" ;;
    *) download_model "checkpoint_vit-l16.pt" "ViT-L/16" ;;
esac

# Download datasets from Hugging Face
echo ""
echo -e "${YELLOW}Downloading datasets from Hugging Face...${NC}"

download_train_dataset() {
    if [ -d "datasets/training_data" ]; then
        echo -e "${GREEN}Training data already exists, skipping...${NC}"
        return
    fi

    echo "Downloading Training data..."
    python -c "
from huggingface_hub import hf_hub_download
import tarfile
import os

tar_path = hf_hub_download(
    repo_id='${HF_REPO}',
    filename='datasets/training_data.tar.gz',
    local_dir='.'
)
print('Extracting...')
with tarfile.open(tar_path, 'r:gz') as tar_ref:
    tar_ref.extractall('datasets')
os.remove(tar_path)
"

    if [ -d "datasets/training_data" ]; then
        echo -e "${GREEN}Downloaded Training data successfully${NC}"
    else
        echo -e "${RED}Failed to download Training data${NC}"
    fi
}

download_eval_dataset() {
    if [ -d "datasets/evaluation_data" ]; then
        echo -e "${GREEN}Evaluation data already exists, skipping...${NC}"
        return
    fi

    echo "Downloading Evaluation data..."
    python -c "
from huggingface_hub import hf_hub_download
import tarfile
import os

tar_path = hf_hub_download(
    repo_id='${HF_REPO}',
    filename='datasets/evaluation_data.tar.gz',
    local_dir='.'
)
print('Extracting...')
with tarfile.open(tar_path, 'r:gz') as tar_ref:
    tar_ref.extractall('datasets')
os.remove(tar_path)
"

    if [ -d "datasets/evaluation_data" ]; then
        echo -e "${GREEN}Downloaded Evaluation data successfully${NC}"
    else
        echo -e "${RED}Failed to download Evaluation data${NC}"
    fi
}

echo ""
echo "Select datasets to download:"
echo "  1) Evaluation only"
echo "  2) Training only"
echo "  3) All datasets"
echo "  4) Skip"
echo ""
read -p "Enter choice [1-4] (default: 4): " dataset_choice
dataset_choice=${dataset_choice:-4}

case $dataset_choice in
    1) download_eval_dataset ;;
    2) download_train_dataset ;;
    3)
        download_eval_dataset
        download_train_dataset
        ;;
    4) echo -e "${YELLOW}Skipping dataset download${NC}" ;;
    *) echo -e "${YELLOW}Skipping dataset download${NC}" ;;
esac

# Verify
echo ""
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "
import torch
import timm
print(f'PyTorch: {torch.__version__}')
print(f'timm: {timm.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"

echo ""
echo "=============================================="
echo -e "${GREEN}  Installation Complete!${NC}"
echo "=============================================="
echo ""
echo "Activate environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Usage:"
echo "  ./scripts/demo.sh                    # Run demo"
echo "  ./scripts/eval.sh                    # Run evaluation"
echo "  ./scripts/train.sh                   # Train model"
echo "  python src/data/download.py          # Download datasets"
echo ""
