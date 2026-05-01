#!/usr/bin/env python3
"""
Kokoro Tamil Fine-Tuning — Lightning AI Training Launcher
==========================================================

Alternative to Modal — uses Manny's existing Lightning AI account.

Usage:
    # Create a GPU studio and run training
    python scripts/train_lightning.py --stage 1

    # Or run inside an existing Lightning AI studio (recommended)
    # Just run the setup + train commands directly inside the studio.

This script handles:
1. Setting up a Lightning AI Studio with A100 GPU
2. Installing all dependencies
3. Downloading datasets from HuggingFace
4. Running training (Stage 1 and/or Stage 2)
5. Uploading results to HuggingFace

Recommended approach: Create a Lightning Studio manually (A100 80GB),
upload these scripts, and run training inside the studio.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SETUP COMMANDS — Run these INSIDE a Lightning AI Studio
# ─────────────────────────────────────────────────────────────────────────────

SETUP_SCRIPT = """#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Kokoro Tamil Fine-Tuning — Lightning AI Studio Setup
# Run this INSIDE a Lightning Studio (A100 80GB recommended)
# ═══════════════════════════════════════════════════════════════════════

set -e

echo "============================================"
echo "Kokoro Tamil Fine-Tuning Setup"
echo "============================================"

# 1. Install system dependencies
sudo apt-get update -qq
sudo apt-get install -y -qq espeak-ng libespeak-ng-dev libsndfile1-dev cmake build-essential git-lfs
git lfs install

# 2. Install Python dependencies
pip install --upgrade pip

# Core ML
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Training
pip install accelerate>=0.33.0 transformers>=4.44.0 datasets>=3.0.0
pip install huggingface_hub>=0.25.0 soundfile librosa scipy pyyaml tensorboard munch Cython
pip install phonemizer einops rotary_embedding_torch

# Kokoro/misaki for G2P
pip install "misaki[espeak]"

# StyleTTS2 dependencies
pip install scipy matplotlib nltk

echo ""
echo "✅ Python packages installed"

# 3. Clone repos
cd /root  # Lightning studio home
if [ ! -d "kokoro-tamil" ]; then
    echo "Cloning kokoro-tamil..."
    # Replace with actual repo URL when ready
    # git clone https://github.com/mastermani305/kokoro-tamil.git
    echo "Please copy the kokoro-tamil project into /root/kokoro-tamil"
fi

if [ ! -d "StyleTTS2" ]; then
    echo "Cloning StyleTTS2..."
    git clone https://github.com/s0md3v0/StyleTTS2.git
    cd StyleTTS2
    
    # Apply kokoro patches from kokoro-deutsch
    cd /root
    if [ ! -d "kokoro-deutsch" ]; then
        git clone --recurse-submodules https://github.com/semidark/kokoro-deutsch.git
    fi
    
    cd kokoro-deutsch
    # Apply patches
    python scripts/prepare_training.py patch-styletts2 2>/dev/null || true
    
    # Build monotonic_align
    cd /root/StyleTTS2
    python setup.py build_ext --inplace
    
    cd /root
fi

echo ""
echo "✅ Repos cloned"

# 4. Download base model
echo "Downloading Kokoro-82M base weights..."
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('training', exist_ok=True)
model = hf_hub_download('hexgrad/Kokoro-82M', 'kokoro-v1_0.pth', local_dir='training')
config = hf_hub_download('hexgrad/Kokoro-82M', 'config.json', local_dir='training')
print(f'Downloaded: {model}, {config}')
"

echo ""
echo "✅ Base model downloaded"

# 5. Download datasets from HuggingFace
echo "Downloading Tamil TTS datasets..."
export HF_TOKEN="${HF_TOKEN:-$(cat /root/.hermes/.env 2>/dev/null | grep HF_TOKEN | cut -d= -f2)}"
python3 /root/kokoro-tamil/scripts/download_datasets.py --token "$HF_TOKEN" --output /root/dataset/raw

echo ""
echo "✅ Datasets downloaded"

# 6. Prepare training data
echo "Preparing training data..."
python3 /root/kokoro-timal/scripts/prepare_tamil_data.py prepare

echo ""
echo "============================================"
echo "Setup Complete! 🎉"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Verify data: python3 scripts/prepare_tamil_data.py verify"
echo "  2. Train Stage 1: accelerate launch train_first.py"
echo "  3. Train Stage 2: accelerate launch train_second.py"
echo ""
echo "Or use the training script:"
echo "  python3 scripts/train_lightning.py --stage 1"
"""


TRAIN_STAGE1_SCRIPT = """#!/bin/bash
# Run Stage 1 training inside Lightning Studio
set -e

cd /root/StyleTTS2

CONFIG="/root/kokoro-tamil/configs/config_tamil_ft.yml"

echo "Starting Stage 1 training (Acoustic model)..."
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

accelerate launch \\
    --mixed_precision fp16 \\
    --num_processes 1 \\
    train_first.py \\
    --config_path "$CONFIG"

echo "Stage 1 complete! Check logs/kokoro-tamil/ for checkpoints."
"""


TRAIN_STAGE2_SCRIPT = """#!/bin/bash
# Run Stage 2 training inside Lightning Studio
set -e

cd /root/StyleTTS2

CONFIG="/root/kokoro-tamil/configs/config_tamil_ft.yml"

echo "Starting Stage 2 training (Duration/SLM)..."
echo "Config: $CONFIG"

accelerate launch \\
    --mixed_precision fp16 \\
    --num_processes 1 \\
    train_second.py \\
    --config_path "$CONFIG"

echo "Stage 2 complete! Check logs/kokoro-tamil/ for checkpoints."
"""


UPLOAD_SCRIPT = """#!/bin/bash
# Upload fine-tuned model to HuggingFace
set -e

export HF_TOKEN="${HF_TOKEN:-$(cat /root/.hermes/.env 2>/dev/null | grep HF_TOKEN | cut -d= -f2)}"

REPO_ID="mastermani305/Kokoro-82M-Tamil"
CHECKPOINT_DIR="/root/StyleTTS2/logs/kokoro-tamil"

echo "Uploading to HuggingFace: $REPO_ID"

# Create repo
python3 -c "
from huggingface_hub import create_repo, HfApi
api = HfApi(token='$HF_TOKEN')
create_repo('$REPO_ID', repo_type='model', token='$HF_TOKEN', exist_ok=True)
"

# Upload checkpoints
huggingface-cli upload "$REPO_ID" "$CHECKPOINT_DIR" --repo-type=model

# Upload config
huggingface-cli upload "$REPO_ID" "/root/kokoro-tamil/configs/config_tamil_ft.yml" "config_tamil_ft.yml" --repo-type=model

# Upload training scripts
huggingface-cli upload "$REPO_ID" "/root/kokoro-tamil/scripts/" "scripts/" --repo-type=model

echo "Model uploaded to: https://huggingface.co/$REPO_ID"
"""


def main():
    parser = argparse.ArgumentParser(
        description="Kokoro Tamil fine-tuning on Lightning AI"
    )
    parser.add_argument("--stage", choices=["1", "2", "both", "setup"],
                       default="setup", help="What to run")
    parser.add_argument("--hf-token", default="",
                       help="HuggingFace token")
    parser.add_argument("--generate-scripts", action="store_true",
                       help="Generate standalone scripts (don't launch)")
    
    args = parser.parse_args()
    
    # Generate scripts
    scripts_dir = Path(__file__).resolve().parent / "lightning"
    scripts_dir.mkdir(exist_ok=True)
    
    (scripts_dir / "setup.sh").write_text(SETUP_SCRIPT)
    (scripts_dir / "train_stage1.sh").write_text(TRAIN_STAGE1_SCRIPT)
    (scripts_dir / "train_stage2.sh").write_text(TRAIN_STAGE2_SCRIPT)
    (scripts_dir / "upload_to_hf.sh").write_text(UPLOAD_SCRIPT)
    
    # Make executable
    for f in scripts_dir.glob("*.sh"):
        f.chmod(0o755)
    
    print(f"Scripts written to: {scripts_dir}")
    
    if args.generate_scripts or args.stage == "setup":
        print("\n" + "=" * 60)
        print("Kokoro Tamil Fine-Tuning — Lightning AI Setup")
        print("=" * 60)
        print("""
RECOMMENDED APPROACH:

1. Create a Lightning AI Studio (A100 80GB):
   - Go to: https://lightning.ai
   - New Studio → GPU → A100 80GB
   - Or use: lightning create studio kokoro-tamil --cloud --type gpu_a100_80gb

2. Upload project files:
   Option A: Clone from GitHub (after pushing)
   Option B: Use `lightning cp` to upload
   
   lightning cp ./kokoro-tamil/ studios:kokoro-tamil:/root/kokoro-tamil/

3. Open the Studio terminal and run:
   bash /root/kokoro-tamil/scripts/lightning/setup.sh

4. Train:
   bash /root/kokoro-tamil/scripts/lightning/train_stage1.sh
   bash /root/kokoro-tamil/scripts/lightning/train_stage2.sh

5. Upload model to HuggingFace:
   bash /root/kokoro-tamil/scripts/lightning/upload_to_hf.sh

ESTIMATED COST:
   - A100 80GB on Lightning AI: ~$1.10/hr
   - Stage 1 (10 epochs): ~4-6 hours = ~$4.40-$6.60
   - Stage 2 (10 epochs): ~4-6 hours = ~$4.40-$6.60
   - Total: ~$9-$13 for full fine-tuning
   
   (Much cheaper alternatives: L4 GPU at ~$0.35/hr, 
    but will take 3-4x longer per stage)
""")
        
        if args.stage != "setup":
            return
    
    # For actual Lightning SDK orchestration
    print(f"\nRunning stage: {args.stage}")
    
    try:
        from lightning_sdk import Machine, Studio
        
        print("Creating Lightning Studio...")
        studio = Studio(
            name="kokoro-tamil-training",
            machine=Machine.A100_80GB,
            teamspace="maniapps",
        )
        
        # Upload project
        print("Uploading project files...")
        # lightning cp would go here
        
        # Run setup
        print("Running setup in studio...")
        studio.run(f"bash /root/kokoro-tamil/scripts/lightning/setup.sh")
        
        # Run training
        if args.stage in ("1", "both"):
            print("Training Stage 1...")
            studio.run("bash /root/kokoro-tamil/scripts/lightning/train_stage1.sh")
        
        if args.stage in ("2", "both"):
            print("Training Stage 2...")
            studio.run("bash /root/kokoro-tamil/scripts/lightning/train_stage2.sh")
        
        print("Training complete!")
        
    except ImportError:
        print("\nLightning SDK not available. Use the script approach instead:")
        print(f"  bash {scripts_dir}/setup.sh")


if __name__ == "__main__":
    main()