# Deep Aerial Image Matching

<p align="center">
  <img src="assets/overview.png" width="400">
</p>

Official PyTorch implementation of:

> **A Two-Stream Symmetric Network with Bidirectional Ensemble for Aerial Image Matching**
> J.-H. Park, W.-J Nam and S.-W Lee
> *Remote Sensing*, 2020, Vol. 12, No. 6, pp. 465
> [[Paper](https://doi.org/10.3390/rs12030465)] [[arXiv](https://arxiv.org/abs/2002.01325)]

## 🎁 Updates

**2025-12-27 (DeepAerialNet v2.0)**
- **DINOv3 ViT backbone**: Added `dinov3` (ViT-Large) as a new backbone option
- **LoFTR-style cross-attention correlation**: New correlation type with transformer cross-attention and 2D positional encoding (`--correlation-type cross_attention`)
- **CoordConv regression**: Optional normalized coordinate channels for regression head (`add_coord=True`)

**2025-12-26**
- Migrated pretrained models and datasets to [Hugging Face Hub](https://huggingface.co/jaehyunnn/DeepAerialMatching)
- Replaced `pretrainedmodels` with `timm` library for modern backbone support
- Added `uv` support for fast environment setup

## Installation

```bash
# Quick install with uv (recommended)
./install.sh

# Or manual installation
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install -e ".[download]"
```

## Pretrained Models

Models are hosted on [Hugging Face Hub](https://huggingface.co/jaehyunnn/DeepAerialMatching).

| Backbone | PCK@0.05 | PCK@0.03 | PCK@0.01 | Download |
|----------|----------|----------|----------|----------|
| ResNet101 | 93.8% | 82.5% | 35.1% | [checkpoint_resnet101.pt](https://huggingface.co/jaehyunnn/DeepAerialMatching/resolve/main/checkpoints/checkpoint_resnet101.pt) |
| ResNeXt101 | 94.6% | 85.9% | 43.2% | [checkpoint_resnext101.pt](https://huggingface.co/jaehyunnn/DeepAerialMatching/resolve/main/checkpoints/checkpoint_resnext101.pt) |
| DenseNet169 | 95.6% | 88.4% | 44.0% | [checkpoint_densenet169.pt](https://huggingface.co/jaehyunnn/DeepAerialMatching/resolve/main/checkpoints/checkpoint_densenet169.pt) |
| **SE-ResNeXt101** | **97.1%** | **91.1%** | **48.0%** | [checkpoint_seresnext101.pt](https://huggingface.co/jaehyunnn/DeepAerialMatching/resolve/main/checkpoints/checkpoint_seresnext101.pt) |

**Download via Python:**
```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="jaehyunnn/DeepAerialMatching",
    filename="checkpoints/checkpoint_seresnext101.pt",
    local_dir="."
)
```

## Datasets

Datasets are also hosted on [Hugging Face Hub](https://huggingface.co/jaehyunnn/DeepAerialMatching).

| Dataset | Description | Download |
|---------|-------------|----------|
| evaluation_data | Evaluation benchmark (500 pairs) | [evaluation_data.tar.gz](https://huggingface.co/jaehyunnn/DeepAerialMatching/resolve/main/datasets/evaluation_data.tar.gz) |
| training_data | Training data (18K pairs with CSV) | [training_data.tar.gz](https://huggingface.co/jaehyunnn/DeepAerialMatching/resolve/main/datasets/training_data.tar.gz) |

**Download via Python:**
```python
from src.data.download import download_eval, download_train

download_eval()   # Download evaluation dataset
download_train()  # Download training dataset
```

**Or via CLI:**
```bash
python src/data/download.py --eval   # Evaluation only
python src/data/download.py --train  # Training only
python src/data/download.py          # All datasets
```

## Usage

```bash
# Demo
./scripts/demo.sh
./scripts/demo.sh --backbone resnet101 --model checkpoints/checkpoint_resnet101.pt
./scripts/demo.sh --backbone dinov3 --correlation-type cross_attention

# Evaluation
./scripts/eval.sh
./scripts/eval.sh --backbone resnet101 --model checkpoints/checkpoint_resnet101.pt

# Training
./scripts/train.sh
./scripts/train.sh --backbone resnet101 --num-epochs 50
./scripts/train.sh --backbone dinov3 --freeze-backbone --correlation-type cross_attention
```

### Available Options

| Option | Values | Description |
|--------|--------|-------------|
| `--backbone` | `resnet101`, `resnext101`, `se_resnext101`, `densenet169`, `dinov3` | Feature extraction backbone |
| `--correlation-type` | `dot`, `cross_attention` | Correlation method (dot: simple dot product, cross_attention: LoFTR-style) |
| `--freeze-backbone` | flag | Freeze backbone weights during training |

### Model Architecture

```
src/models/
├── backbone.py      # FeatureExtraction (CNN/ViT backbones)
├── correlation.py   # FeatureCorrelation, CrossAttentionCorrelation
├── layers.py        # FeatureL2Norm, FeatureRegression
├── aerial_net.py    # AerialNetSingleStream, AerialNetTwoStream
└── loss.py          # TransformedGridLoss
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- timm >= 1.0

## Citation

```bibtex
@article{park2020aerial,
  title={A Two-Stream Symmetric Network with Bidirectional Ensemble for Aerial Image Matching},
  author={Park, Jae-Hyun and Nam, Woo-Jeoung and Lee, Seong-Whan},
  journal={Remote Sensing},
  volume={12},
  number={3},
  pages={465},
  year={2020},
  publisher={MDPI}
}
```