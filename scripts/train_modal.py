#!/usr/bin/env python3
"""
Kokoro Tamil Fine-Tuning — Modal.com Training Launcher
========================================================

Runs the full 2-stage Kokoro TTS fine-tuning pipeline on Modal GPUs.

Usage:
    # Dry run (checks setup, no GPU)
    python scripts/train_modal.py --dry-run

    # Stage 1 only (acoustic/prosody — ~4-6 hrs on A100)
    python scripts/train_modal.py --stage 1

    # Stage 2 only (duration/SLM — ~4-6 hrs on A100)
    python scripts/train_modal.py --stage 2

    # Both stages
    python scripts/train_modal.py --stage both

    # Resume from checkpoint
    python scripts/train_modal.py --stage 1 --resume-epoch 5

    # Upload results to HuggingFace
    python scripts/train_modal.py --upload-to-hf

Requirements:
    - Modal CLI configured (modal profile list)
    - HF_TOKEN in environment or --hf-token flag
    - Sufficient Modal credits
"""

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Modal App Definition
# ─────────────────────────────────────────────────────────────────────────────

MODAL_APP_CODE = '''
import modal
import os
import subprocess
import json
from pathlib import Path

app = modal.App("kokoro-tamil-training")

# — Volume for persistent data across runs —
vol = modal.Volume.from_name("kokoro-tamil-data", create_if_missing=True)
MODEL_DIR = "/root/data"

# — Base image with all dependencies —
base_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
    .apt_install(
        "python3-dev", "python3-pip", "git", "git-lfs", "espeak-ng",
        "libsndfile1-dev", "libespeak-ng-dev", "cmake", "build-essential",
    )
    .pip_install(
        "torch==2.4.0+cu124", "torchaudio==2.4.0+cu124",
        "accelerate>=0.33.0", "transformers>=4.44.0", "datasets>=3.0.0",
        "huggingface_hub>=0.25.0", "soundfile", "librosa", "scipy",
        "pyyaml", "tensorboard", "munch", "phonemizer", "Cython",
        "misaki[espeak]", "einops", "rotary_embedding_torch",
        index_urls={"torch": "https://download.pytorch.org/whl/cu124"},
    )
    .run_commands(
        "git lfs install",
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download datasets
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image=base_image,
    volumes={MODEL_DIR: vol},
    timeout=7200,  # 2 hours
    memory=32768,  # 32 GB
)
def download_datasets(hf_token: str):
    """Download all Tamil TTS datasets from HuggingFace."""
    import os
    from huggingface_hub import snapshot_download, login
    
    login(token=hf_token)
    os.environ["HF_TOKEN"] = hf_token
    
    datasets = {
        "kavi-ps-manual-10hr-original": "mastermani305/kavi-ps-manual-10hr-original",
        "ps-transcribed-ps1": ("mastermani305/ps-transcribed", "ps-1"),
        "ps-transcribed-ps2": ("mastermani305/ps-transcribed", "ps-2"),
        "tts-1000-v1": "mastermani305/tts-1000-v1",
    }
    
    for name, spec in datasets.items():
        out_dir = f"{MODEL_DIR}/raw/{name}"
        if os.path.exists(out_dir) and os.listdir(out_dir):
            print(f"SKIP (already downloaded): {name}")
            continue
        
        print(f"Downloading: {name}...")
        if isinstance(spec, tuple):
            repo_id, config = spec
            snapshot_download(
                repo_id, repo_type="dataset", token=hf_token,
                local_dir=out_dir,
            )
        else:
            snapshot_download(
                spec, repo_type="dataset", token=hf_token,
                local_dir=out_dir,
            )
    
    # Also download base Kokoro model
    from huggingface_hub import hf_hub_download
    model_dir = f"{MODEL_DIR}/base_model"
    os.makedirs(model_dir, exist_ok=True)
    
    for filename in ["kokoro-v1_0.pth", "config.json"]:
        path = hf_hub_download("hexgrad/Kokoro-82M", filename, local_dir=model_dir)
        print(f"Downloaded: {path}")
    
    vol.commit()
    print("Dataset download complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Prepare training data
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image=base_image,
    volumes={MODEL_DIR: vol},
    timeout=1800,  # 30 min
    memory=16384,
)
def prepare_data():
    """Extract WAVs, phonemize, generate train/val lists."""
    # This runs the prepare_tamil_data.py script
    result = subprocess.run(
        ["python3", f"{MODEL_DIR}/kokoro-tamil/scripts/prepare_tamil_data.py", "prepare"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError("Data preparation failed")
    
    vol.commit()
    print("Data preparation complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Train Stage 1 (Acoustic model)
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image=base_image,
    gpu=modal.gpu.A100(size="80GB"),
    volumes={MODEL_DIR: vol},
    timeout=86400,  # 24 hours max
    memory=65536,   # 64 GB
    cpu=8,
)
def train_stage1(resume_epoch: int = 0):
    """Fine-tune Kokoro Stage 1 (acoustic/prosody model)."""
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Clone StyleTTS2 + kokoro-deutsch patches
    sty_d = f"{MODEL_DIR}/StyleTTS2"
    if not os.path.exists(sty_d):
        os.chdir(MODEL_DIR)
        subprocess.run(["git", "clone", "https://github.com/s0md3v0/StyleTTS2.git", sty_d], check=True)
        subprocess.run(["git", "clone", "https://github.com/semidark/kokoro-deutsch.git", f"{MODEL_DIR}/kokoro-deutsch"], check=True)
    
    # Apply kokoro patches
    os.chdir(sty_d)
    
    config_path = f"{MODEL_DIR}/kokoro-tamil/configs/config_tamil_ft.yml"
    
    cmd = [
        "accelerate", "launch", "--mixed_precision", "fp16",
        f"{sty_d}/train_first.py",
        "--config_path", config_path,
    ]
    
    if resume_epoch > 0:
        cmd.extend(["--resume_epoch", str(resume_epoch)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 training failed with code {result.returncode}")
    
    vol.commit()
    print("Stage 1 training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Train Stage 2 (Duration/SLM)
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image=base_image,
    gpu=modal.gpu.A100(size="80GB"),
    volumes={MODEL_DIR: vol},
    timeout=86400,
    memory=65536,
    cpu=8,
)
def train_stage2(resume_epoch: int = 0):
    """Fine-tune Kokoro Stage 2 (duration predictor + SLM adversarial)."""
    config_path = f"{MODEL_DIR}/kokoro-tamil/configs/config_tamil_ft.yml"
    sty_d = f"{MODEL_DIR}/StyleTTS2"
    
    cmd = [
        "accelerate", "launch", "--mixed_precision", "fp16",
        f"{sty_d}/train_second.py",
        "--config_path", config_path,
    ]
    
    if resume_epoch > 0:
        cmd.extend(["--resume_epoch", str(resume_epoch)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Stage 2 training failed with code {result.returncode}")
    
    vol.commit()
    print("Stage 2 training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Extract voicepack & upload to HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image=base_image,
    volumes={MODEL_DIR: vol},
    timeout=3600,
    memory=16384,
)
def upload_to_hf(hf_token: str, stage2_epoch: int = 10):
    """Extract voicepack and upload fine-tuned model to HuggingFace."""
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi(token=hf_token)
    repo_id = "mastermani305/Kokoro-82M-Tamil"
    
    # Create repo if not exists
    try:
        create_repo(repo_id, repo_type="model", token=hf_token, exist_ok=True)
    except Exception:
        pass
    
    # Upload checkpoint
    checkpoint_dir = f"{MODEL_DIR}/StyleTTS2/logs/kokoro-tamil"
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        token=hf_token,
    )
    
    # Upload config
    api.upload_file(
        path_or_fileobj=f"{MODEL_DIR}/kokoro-tamil/configs/config_tamil_ft.yml",
        path_in_repo="config_tamil_ft.yml",
        repo_id=repo_id,
        token=hf_token,
    )
    
    print(f"Model uploaded to: https://huggingface.co/{repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Test inference
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image=base_image,
    gpu=modal.gpu.A100(size="80GB"),
    volumes={MODEL_DIR: vol},
    timeout=1800,
    memory=32768,
)
def test_inference(hf_token: str):
    """Generate test samples from fine-tuned model."""
    import torch
    
    # Download base Kokoro for voicepack extraction
    from huggingface_hub import hf_hub_download
    model_dir = f"{MODEL_DIR}/base_model"
    
    # Run test inference using the fine-tuned checkpoint
    checkpoint_path = f"{MODEL_DIR}/StyleTTS2/logs/kokoro-tamil/epoch_2nd_00010.pth"
    output_dir = f"{MODEL_DIR}/test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use Kokoro's inference pipeline
    subprocess.run([
        "python3", f"{MODEL_DIR}/kokoro-tamil/scripts/test_inference.py",
        "--checkpoint", checkpoint_path,
        "--output-dir", output_dir,
    ], check=True)
    
    vol.commit()
    print(f"Test outputs saved to: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(stage: str = "both", hf_token: str = "", upload: bool = False):
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HF_TOKEN required (set env var or pass --hf-token)")
    
    print("=" * 60)
    print("Kokoro-82M Tamil Fine-Tuning Pipeline")
    print("=" * 60)
    
    # Step 1: Download data
    print("\\n[1/5] Downloading datasets...")
    download_datasets.remote(hf_token=hf_token)
    
    # Step 2: Prepare data
    if stage in ("1", "2", "both"):
        print("\\n[2/5] Preparing training data...")
        prepare_data.remote()
    
    # Step 3: Stage 1 training
    if stage in ("1", "both"):
        print("\\n[3/5] Training Stage 1 (acoustic model)...")
        train_stage1.remote()
    
    # Step 4: Stage 2 training
    if stage in ("2", "both"):
        print("\\n[4/5] Training Stage 2 (duration/SLM)...")
        train_stage2.remote()
    
    # Step 5: Upload
    if upload:
        print("\\n[5/5] Uploading to HuggingFace...")
        upload_to_hf.remote(hf_token=hf_token)
    
    print("\\nDone! 🎉")
'''


def write_modal_app():
    """Write the Modal app file."""
    app_path = ROOT_DIR / "scripts" / "train_modal_app.py"
    app_path.write_text(MODAL_APP_CODE)
    print(f"Wrote: {app_path}")
    return app_path


def check_modal_setup():
    """Verify Modal CLI is configured."""
    result = subprocess.run(
        ["modal", "profile", "list"], 
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: Modal CLI not configured")
        print("Run: modal setup")
        return False
    
    profiles = result.stdout.strip()
    if not profiles or len(profiles.split('\n')) < 2:
        print("WARNING: No Modal profile configured")
        print("Run: modal setup")
        return False
    
    print(f"Modal profiles: {profiles}")
    return True


def estimate_cost(stage="both", hours_per_stage=6, gpu_type="A100-80GB"):
    """Estimate training cost on Modal."""
    # Modal approx pricing (as of 2024)
    prices = {
        "A100-80GB": 3.92,   # $/hr
        "A100-40GB": 2.28,
        "L4": 0.64,
    }
    
    rate = prices.get(gpu_type, 3.92)
    
    stages = 2 if stage == "both" else 1
    total_hours = stages * hours_per_stage
    total_cost = total_hours * rate
    
    print(f"\n{'='*50}")
    print(f"Estimated Training Cost (Modal)")
    print(f"{'='*50}")
    print(f"GPU: {gpu_type}")
    print(f"Rate: ${rate:.2f}/hr")
    print(f"Stages: {stages}")
    print(f"Hours per stage: ~{hours_per_stage}")
    print(f"Total hours: ~{total_hours}")
    print(f"Estimated cost: ~${total_cost:.2f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch Kokoro Tamil fine-tuning on Modal.com"
    )
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both",
                       help="Training stage to run")
    parser.add_argument("--hf-token", default="", 
                       help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Check setup without launching")
    parser.add_argument("--upload-to-hf", action="store_true",
                       help="Upload model to HuggingFace after training")
    parser.add_argument("--cost-only", action="store_true",
                       help="Show cost estimate and exit")
    parser.add_argument("--gpu", choices=["A100-80GB", "A100-40GB", "L4"],
                       default="A100-80GB", help="GPU type")
    
    args = parser.parse_args()
    
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    
    # Estimate cost
    estimate_cost(stage=args.stage, gpu_type=args.gpu)
    
    if args.cost_only:
        return
    
    # Check Modal setup
    if not check_modal_setup():
        print("\nTo set up Modal:")
        print("  1. Create account: https://modal.com")
        print("  2. Run: modal setup")
        sys.exit(1)
    
    # Write Modal app
    app_path = write_modal_app()
    
    if args.dry_run:
        print("\nDRY RUN — Modal app written to:")
        print(f"  {app_path}")
        print("\nTo launch training:")
        print(f"  modal run {app_path} --stage {args.stage} --hf-token YOUR_TOKEN")
        return
    
    if not hf_token:
        print("ERROR: HF_TOKEN required. Set env var or pass --hf-token")
        sys.exit(1)
    
    # Launch on Modal
    cmd = [
        "modal", "run", str(app_path),
        "--stage", args.stage,
        "--hf-token", hf_token,
    ]
    if args.upload_to_hf:
        cmd.append("--upload")
    
    print(f"\nLaunching training on Modal: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()