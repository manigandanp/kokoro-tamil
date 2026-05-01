#!/usr/bin/env python3
"""
Download Tamil TTS datasets from HuggingFace.

Usage:
    python scripts/download_datasets.py --token $HF_TOKEN --output ./dataset
"""

import argparse
import os
import sys
from pathlib import Path


def download_dataset(repo_id, output_dir, token=None, configs=None):
    """Download a dataset from HuggingFace."""
    from datasets import load_dataset

    kwargs = {"path": output_dir}
    if token:
        kwargs["token"] = token

    if configs:
        for config in configs:
            print(f"  Downloading config: {config}...")
            ds = load_dataset(repo_id, config, token=token)
            save_path = Path(output_dir) / repo_id.replace("/", "_") / config
            ds.save_to_disk(str(save_path))
            print(f"  Saved to {save_path}")
    else:
        print(f"  Downloading default config...")
        ds = load_dataset(repo_id, token=token)
        save_path = Path(output_dir) / repo_id.replace("/", "_")
        ds.save_to_disk(str(save_path))
        print(f"  Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Download Tamil TTS datasets from HuggingFace")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    parser.add_argument("--output", default="./dataset/raw", help="Output directory")
    parser.add_argument("--skip-large", action="store_true", help="Skip ps-transcribed (18GB)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_TOKEN"] = args.token

    # 1. kavi-ps-manual-10hr-original (PRIMARY - manually verified)
    print("\n=== Downloading kavi-ps-manual-10hr-original (8K verified) ===")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            "mastermani305/kavi-ps-manual-10hr-original",
            repo_type="dataset",
            token=args.token,
            local_dir=str(output_dir / "kavi-ps-manual-10hr-original"),
        )
        print(f"  Downloaded to: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. ps-transcribed (LARGEST - 21K samples with timestamps)
    if not args.skip_large:
        print("\n=== Downloading ps-transcribed (~21K samples, 18GB) ===")
        for config in ["ps-1", "ps-2", "ps-3", "ps-4"]:
            print(f"  Config: {config}...")
            try:
                path = snapshot_download(
                    f"mastermani305/ps-transcribed",
                    repo_type="dataset",
                    token=args.token,
                    local_dir=str(output_dir / "ps-transcribed"),
                    allow_patterns=[f"{config}/*"],
                )
                print(f"  Downloaded config {config}")
            except Exception as e:
                print(f"  ERROR ({config}): {e}")

    # 3. tts-1000-v1 (PUBLIC - smaller supplementary)
    print("\n=== Downloading tts-1000-v1 (1K, 193MB) ===")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            "mastermani305/tts-1000-v1",
            repo_type="dataset",
            token=args.token,
            local_dir=str(output_dir / "tts-1000-v1"),
        )
        print(f"  Downloaded to: {path}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n=== Download complete! ===")
    print(f"Data saved to: {output_dir}")


if __name__ == "__main__":
    main()