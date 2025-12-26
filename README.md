# Deep Aerial Image Matching

<p align="center">
  <img src="https://www.mdpi.com/remotesensing/remotesensing-12-00465/article_deploy/html/images/remotesensing-12-00465-ag-550.jpg" width="400">
</p>

Official PyTorch implementation of:

> **A Two-Stream Symmetric Network with Bidirectional Ensemble for Aerial Image Matching**
> J.-H. Park, W.-J Nam and S.-W Lee
> *Remote Sensing*, 2020, Vol. 12, No. 6, pp. 465
> [[Paper](https://doi.org/10.3390/rs12030465)] [[arXiv](https://arxiv.org/abs/2002.01325)]

## 🎁 Updates

**2025-12-27**
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
python src/cli/demo.py

# Evaluation
./scripts/eval.sh --model checkpoints/checkpoint_seresnext101.pt --cnn se_resnext101

# Training
python src/cli/train.py --feature-extraction-cnn se_resnext101
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