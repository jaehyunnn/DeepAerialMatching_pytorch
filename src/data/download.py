"""Dataset download utilities using Hugging Face Hub."""
from __future__ import annotations

import os
import tarfile
import zipfile
from typing import Optional

from huggingface_hub import hf_hub_download

# Hugging Face repository
HF_REPO = "jaehyunnn/DeepAerialMatching"


def _download_and_extract_zip(filename: str, dest_dir: str, extract_name: Optional[str] = None):
    """Download a zip file from Hugging Face and extract it.

    Args:
        filename: Name of the file in the HF repo (e.g., 'datasets/current.zip')
        dest_dir: Directory to extract files to
        extract_name: Optional name for extraction folder check
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Check if already extracted
    check_path = os.path.join(dest_dir, extract_name) if extract_name else dest_dir
    if extract_name and os.path.exists(check_path):
        print(f"{extract_name} already exists, skipping...")
        return

    print(f"Downloading {filename}...")

    # Download from Hugging Face
    zip_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="model",
        local_dir=".",
    )

    # Extract
    print(f"Extracting to {dest_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    # Remove zip file after extraction
    os.remove(zip_path)
    print(f"Done: {filename}")


def _download_and_extract_tar(filename: str, dest_dir: str):
    """Download a tar.gz file from Hugging Face and extract it.

    Args:
        filename: Name of the file in the HF repo (e.g., 'datasets/evaluation_data.tar.gz')
        dest_dir: Directory to extract files to
    """
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Downloading {filename}...")

    # Download from Hugging Face
    tar_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="model",
        local_dir=".",
    )

    # Extract
    print(f"Extracting to {dest_dir}...")
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(dest_dir)

    # Remove tar file after extraction
    os.remove(tar_path)
    print(f"Done: {filename}")


def download_train(datapath: str = "datasets"):
    """Download training dataset (training_data with current/past images and CSV files).

    Args:
        datapath: Base directory for datasets
    """
    os.makedirs(datapath, exist_ok=True)

    # Check for training_data folder
    dest = os.path.join(datapath, 'training_data')
    if not os.path.exists(dest):
        _download_and_extract_tar(
            filename='datasets/training_data.tar.gz',
            dest_dir=datapath,
        )
    else:
        print("training_data already exists, skipping...")


def download_eval(datapath: str = "datasets"):
    """Download evaluation dataset.

    Args:
        datapath: Base directory for datasets
    """
    os.makedirs(datapath, exist_ok=True)

    # Check for evaluation_data folder
    dest = os.path.join(datapath, 'evaluation_data')
    if not os.path.exists(dest):
        _download_and_extract_tar(
            filename='datasets/evaluation_data.tar.gz',
            dest_dir=datapath,
        )
    else:
        print("evaluation_data already exists, skipping...")


def download_model(backbone: str = "vit-l/16", dest_dir: str = "checkpoints"):
    """Download pretrained model checkpoint.

    Args:
        backbone: Model backbone name (e.g., 'vit-l/16', 'resnet101', 'se_resnext101')
        dest_dir: Directory to save checkpoint
    """
    # Map backbone names to checkpoint filenames
    model_files = {
        "vit-l/16": "checkpoints/checkpoint_vit-l16.pt",
        "resnet101": "checkpoints/checkpoint_resnet101.pt",
        "resnext101": "checkpoints/checkpoint_resnext101.pt",
        "se_resnext101": "checkpoints/checkpoint_seresnext101.pt",
        "densenet169": "checkpoints/checkpoint_densenet169.pt",
    }

    if backbone not in model_files:
        raise ValueError(f"Unknown backbone: {backbone}. Supported: {list(model_files.keys())}")

    os.makedirs(dest_dir, exist_ok=True)
    filename = model_files[backbone]
    dest_path = os.path.join(dest_dir, os.path.basename(filename))

    # Check if already downloaded
    if os.path.exists(dest_path):
        print(f"{os.path.basename(filename)} already exists, skipping...")
        return dest_path

    print(f"Downloading {backbone} checkpoint...")

    # Download from Hugging Face
    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="model",
        local_dir=".",
    )

    print(f"Done: {os.path.basename(filename)}")
    return checkpoint_path


def download_all(datapath: str = "datasets"):
    """Download all datasets (training + evaluation).

    Args:
        datapath: Base directory for datasets
    """
    print("Downloading training dataset...")
    download_train(datapath)

    print("\nDownloading evaluation dataset...")
    download_eval(datapath)

    print("\nAll datasets downloaded successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets and models from Hugging Face")
    parser.add_argument('--datapath', type=str, default='datasets',
                        help='Directory to save datasets')
    parser.add_argument('--train', action='store_true',
                        help='Download only training data')
    parser.add_argument('--eval', action='store_true',
                        help='Download only evaluation data')
    parser.add_argument('--model', type=str, choices=['vit-l/16', 'resnet101', 'resnext101', 'se_resnext101', 'densenet169'],
                        help='Download pretrained model checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoint')
    args = parser.parse_args()

    if args.model:
        download_model(args.model, args.checkpoint_dir)
    elif args.train:
        download_train(args.datapath)
    elif args.eval:
        download_eval(args.datapath)
    else:
        download_all(args.datapath)
