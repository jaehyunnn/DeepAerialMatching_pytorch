#!/usr/bin/env python
"""Upload trained models and datasets to Hugging Face Hub."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from huggingface_hub import HfApi, create_repo

# Configuration
REPO_ID = "jaehyunnn/DeepAerialMatching"

MODELS = {
    "checkpoint_seresnext101.pt": "SE-ResNeXt101 (Best)",
    "checkpoint_resnext101.pt": "ResNeXt101",
    "checkpoint_resnet101.pt": "ResNet101",
    "checkpoint_densenet169.pt": "DenseNet169",
}

DATASETS = {
    "current.zip": "Training data - Current aerial images",
    "past.zip": "Training data - Past aerial images",
    "GoogleEarth_pck.zip": "Evaluation data - GoogleEarth PCK benchmark",
}


def upload_models(api: HfApi):
    """Upload model checkpoints."""
    print("\n=== Uploading Models ===")

    for filename, description in MODELS.items():
        filepath = f"checkpoints/{filename}"
        if not os.path.exists(filepath):
            print(f"Skipping {filename} (not found)")
            continue

        print(f"Uploading {description}...")
        try:
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f"checkpoints/{filename}",
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  Uploaded: {filename}")
        except Exception as e:
            print(f"  Error: {e}")


def upload_datasets(api: HfApi):
    """Upload dataset zip files."""
    print("\n=== Uploading Datasets ===")

    for filename, description in DATASETS.items():
        filepath = f"datasets/{filename}"
        if not os.path.exists(filepath):
            print(f"Skipping {filename} (not found)")
            continue

        print(f"Uploading {description}...")
        try:
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=f"datasets/{filename}",
                repo_id=REPO_ID,
                repo_type="model",
            )
            print(f"  Uploaded: {filename}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Upload to Hugging Face Hub")
    parser.add_argument('--models', action='store_true',
                        help='Upload only models')
    parser.add_argument('--datasets', action='store_true',
                        help='Upload only datasets')
    args = parser.parse_args()

    api = HfApi()

    # Create repo if not exists
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"Note: {e}")

    # Determine what to upload
    if args.models:
        upload_models(api)
    elif args.datasets:
        upload_datasets(api)
    else:
        # Upload both by default
        upload_models(api)
        upload_datasets(api)

    print(f"\nDone! Available at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
