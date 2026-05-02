#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Kokoro Tamil TTS — Studio Restart Recovery Script
# ─────────────────────────────────────────────────────────────────────────────
# Run this AFTER a Studio restart (4-hour free tier reset).
# It will: reinstall deps, recover checkpoints from HF, and resume training.
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "===== Kokoro Tamil TTS — Studio Restart Recovery ====="
echo "Starting at: $(date)"

cd /teamspace/studios/this_studio

# ── Step 1: Reinstall dependencies ──────────────────────────────────────────
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq espeak-ng cmake build-essential sox libsndfile1-dev ffmpeg > /dev/null 2>&1

echo "[1/6] Installing Python dependencies..."
pip install --quiet torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --quiet transformers==4.47.1 'misaki[zh]' phonemizer einops-exts click accelerate meldataset scipy
pip install --quiet huggingface_hub sentencepiece

# ── Step 2: Re-login to HuggingFace ─────────────────────────────────────────
echo "[2/6] Logging into HuggingFace..."
huggingface-cli login --token "$HF_TOKEN" --add-to-credential 2>/dev/null || true

# ── Step 3: Download checkpoint from HuggingFace if local is missing ────────
LOG_DIR="/teamspace/studios/this_studio/training/logs/kokoro-tamil"
mkdir -p "$LOG_DIR"

if [ ! -f "$LOG_DIR"/epoch_1st_*.pth ]; then
    echo "[3/6] No local checkpoint found. Downloading from HuggingFace..."
    python3 -c "
from huggingface_hub import hf_hub_download, list_repo_files
import shutil, os
repo = 'mastermani305/kokoro-tamil-backups'
log_dir = '$LOG_DIR'
files = list_repo_files(repo, repo_type='model')
for f in files:
    if f.startswith('stage1_checkpoints/') and f.endswith('.pth'):
        print(f'Downloading {f}...')
        path = hf_hub_download(repo, f, repo_type='model')
        dest = os.path.join(log_dir, os.path.basename(f))
        shutil.copy2(path, dest)
        print(f'Saved to {dest}')
    elif f.startswith('checkpoints/') and f.endswith('.pth'):
        # Also check old format
        path = hf_hub_download(repo, f, repo_type='model')
        dest = os.path.join(log_dir, os.path.basename(f))
        shutil.copy2(path, dest)
        print(f'Saved to {dest}')
"
else
    echo "[3/6] Local checkpoint found. Skipping download."
fi

# ── Step 4: Find latest checkpoint and update config ────────────────────────
echo "[4/6] Finding latest checkpoint..."
LATEST_CKPT=$(ls -t "$LOG_DIR"/epoch_1st_*.pth 2>/dev/null | head -1)

CONFIG="/teamspace/studios/this_studio/training/config_tamil_ft.yml"

if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from: $LATEST_CKPT"
    python3 -c "
import yaml
with open('$CONFIG') as f:
    config = yaml.safe_load(f)
config['pretrained_model'] = '$LATEST_CKPT'
config['load_only_params'] = False  # Resume optimizer + epoch
with open('$CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f'Config updated: resume from $LATEST_CKPT')
"
else
    echo "No checkpoint found. Starting from base model."
fi

# ── Step 5: Verify data symlink ────────────────────────────────────────────
echo "[5/6] Verifying data..."
cd /teamspace/studios/this_studio/kokoro-deutsch/StyleTTS2
if [ ! -L wav ]; then
    ln -sf /teamspace/studios/this_studio/dataset/audio/kavi wav
    echo "WAV symlink restored"
fi

# ── Step 6: Start training + watchdog ───────────────────────────────────────
echo "[6/6] Starting training..."
nohup python train_first.py -p "$CONFIG" >> /teamspace/studios/this_studio/training/stage1.log 2>&1 &
TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# Start watchdog
nohup bash /teamspace/studios/this_studio/watchdog.sh > /teamspace/studios/this_studio/watchdog.out 2>&1 &
echo "Watchdog PID: $!"

# Wait and verify
sleep 15
echo ""
echo "=== Verification ==="
ps aux | grep train_first | grep -v grep | head -2
echo ""
tail -10 /teamspace/studios/this_studio/training/stage1.log

echo ""
echo "===== Recovery Complete ====="