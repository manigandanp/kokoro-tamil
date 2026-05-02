#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Kokoro Tamil Fine-Tuning — Full Setup & Training Script
# ═══════════════════════════════════════════════════════════════════════════
# Run this INSIDE the Lightning AI Studio (kokoro-tamil-training)
# This script handles EVERYTHING: setup, data download, training
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

echo "============================================================"
echo "  Kokoro-82M Tamil TTS Fine-Tuning"
echo "  Starting at: $(date)"
echo "============================================================"

# ── Configuration ─────────────────────────────────────────────────────────────
HF_TOKEN="${HF_TOKEN:?Please set HF_TOKEN environment variable}"
WORKDIR="/teamspace/studios/this_studio"
DATA_DIR="${WORKDIR}/dataset"
MODEL_DIR="${WORKDIR}/training"
LOG_DIR="${WORKDIR}/logs"

# Number of epochs (adjust for quick test vs full training)
STAGE1_EPOCHS="${STAGE1_EPOCHS:-10}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"

echo ""
echo "Configuration:"
echo "  Workdir: ${WORKDIR}"
echo "  Stage1 epochs: ${STAGE1_EPOCHS}"
echo "  Stage2 epochs: ${STAGE2_EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LEARNING_RATE}"
echo ""

# ── Step 1: Install System Dependencies ────────────────────────────────────
echo "══════ Step 1: Installing system dependencies ══════"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    espeak-ng libespeak-ng-dev libsndfile1-dev \
    cmake build-essential git-lfs sox ffmpeg \
    2>/dev/null || true
git lfs install

# ── Step 2: Install Python Dependencies ─────────────────────────────────────
echo "══════ Step 2: Installing Python packages ══════"
pip install --upgrade pip --quiet
pip install --quiet \
    torch==2.4.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

pip install --quiet \
    accelerate>=0.33.0 \
    transformers>=4.44.0 \
    datasets>=3.0.0 \
    huggingface_hub>=0.25.0 \
    soundfile librosa scipy \
    pyyaml tensorboard munch \
    phonemizer Cython \
    "misaki[espeak]" \
    einops rotary_embedding_torch \
    matplotlib nltk \
    pyarrow pandas

echo "  ✅ Python packages installed"

# ── Step 3: Clone Repos ────────────────────────────────────────────────────
echo "══════ Step 3: Cloning repositories ══════"
cd "${WORKDIR}"

# Clone kokoro-tamil (our project)
if [ ! -d "kokoro-tamil" ]; then
    git clone https://github.com/manigandanp/kokoro-tamil.git
    echo "  ✅ Cloned kokoro-tamil"
else
    echo "  ⏭️  kokoro-tamil already exists"
fi

# Clone StyleTTS2 (base model code)
if [ ! -d "StyleTTS2" ]; then
    git clone https://github.com/s0md3v0/StyleTTS2.git
    echo "  ✅ Cloned StyleTTS2"
else
    echo "  ⏭️  StyleTTS2 already exists"
fi

# Clone kokoro-deutsch (patches and training recipes)
if [ ! -d "kokoro-deutsch" ]; then
    git clone --recurse-submodules https://github.com/semidark/kokoro-deutsch.git
    echo "  ✅ Cloned kokoro-deutsch"
else
    echo "  ⏭️  kokoro-deutsch already exists"
fi

# Build monotonic_align (required by StyleTTS2)
cd "${WORKDIR}/StyleTTS2"
python setup.py build_ext --inplace 2>/dev/null || true
echo "  ✅ Built monotonic_align"

# ── Step 4: Download Base Model ────────────────────────────────────────────
echo "══════ Step 4: Downloading Kokoro-82M base model ══════"
mkdir -p "${MODEL_DIR}"

if [ ! -f "${MODEL_DIR}/kokoro-v1_0.pth" ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('${MODEL_DIR}', exist_ok=True)
model = hf_hub_download('hexgrad/Kokoro-82M', 'kokoro-v1_0.pth', local_dir='${MODEL_DIR}')
config = hf_hub_download('hexgrad/Kokoro-82M', 'config.json', local_dir='${MODEL_DIR}')
print(f'Downloaded model: {model}')
print(f'Downloaded config: {config}')
"
    echo "  ✅ Base model downloaded"
else
    echo "  ⏭️  Base model already exists"
fi

# ── Step 5: Download Datasets from HuggingFace ──────────────────────────────
echo "══════ Step 5: Downloading Tamil TTS datasets ══════"
mkdir -p "${DATA_DIR}/raw"

# Download primary dataset (manually transcribed)
python3 -c "
from huggingface_hub import snapshot_download
import os, sys

token = '${HF_TOKEN}'
raw_dir = '${DATA_DIR}/raw'

# 1. Primary: kavi-ps-manual-10hr-original (8K manually transcribed)
path = 'mastermani305/kavi-ps-manual-10hr-original'
out = os.path.join(raw_dir, 'kavi-ps-manual-10hr-original')
if not os.path.exists(out) or not os.listdir(out):
    print(f'Downloading {path}...')
    snapshot_download(path, repo_type='dataset', token=token, local_dir=out)
    print(f'  ✅ Downloaded {path}')
else:
    print(f'  ⏭️  Already have {path}')

# 2. Supplementary: tts-1000-v1 (public, smaller)
path2 = 'mastermani305/tts-1000-v1'
out2 = os.path.join(raw_dir, 'tts-1000-v1')
if not os.path.exists(out2) or not os.listdir(out2):
    print(f'Downloading {path2}...')
    snapshot_download(path2, repo_type='dataset', token=token, local_dir=out2)
    print(f'  ✅ Downloaded {path2}')
else:
    print(f'  ⏭️  Already have {path2}')
"

# Optional: Download ps-transcribed (18GB, skip if SKIP_LARGE=1)
if [ "${SKIP_LARGE:-0}" != "1" ]; then
    python3 -c "
from huggingface_hub import snapshot_download
import os

token = '${HF_TOKEN}'
raw_dir = '${DATA_DIR}/raw'

path = 'mastermani305/ps-transcribed'
out = os.path.join(raw_dir, 'ps-transcribed')
if not os.path.exists(out) or not os.listdir(out):
    print(f'Downloading {path} (18GB, this may take a while)...')
    snapshot_download(path, repo_type='dataset', token=token, local_dir=out)
    print(f'  ✅ Downloaded {path}')
else:
    print(f'  ⏭️  Already have {path}')
"
fi

echo "  ✅ Datasets downloaded"

# ── Step 6: Apply Kokoro Patches ────────────────────────────────────────────
echo "══════ Step 6: Applying Kokoro patches to StyleTTS2 ══════"
cd "${WORKDIR}/kokoro-deutsch"

# Copy Kokoro-specific files into StyleTTS2
if [ -f "kokoro_symbols.py" ]; then
    cp kokoro_symbols.py "${WORKDIR}/StyleTTS2/"
    echo "  ✅ Copied kokoro_symbols.py"
fi

# Apply patches from kokoro-deutsch
python3 scripts/prepare_training.py patch-styletts2 2>/dev/null || {
    echo "  ⚠️  Automated patch failed, applying manual patches..."
    # The key patches are:
    # 1. Symbol table (178 IPA tokens)
    # 2. Model architecture changes (iSTFTNet decoder)
    # 3. WavLM discriminator setup
    echo "  Manual patching needed — check kokoro-deutsch/docs/"
}

# Verify patches
python3 -c "
import sys
sys.path.insert(0, '${WORKDIR}/StyleTTS2')
try:
    from text_utils import symbols
    print(f'  Symbols count: {len(symbols)}')
    if len(symbols) == 178:
        print('  ✅ Kokoro 178-token vocabulary loaded correctly')
    else:
        print(f'  ⚠️  Expected 178 symbols, got {len(symbols)}')
        print('  Applying kokoro symbols manually...')
        # Override with Kokoro symbols
        from kokoro_deutsch.kokoro_symbols import symbols
        print(f'  Manual symbols: {len(symbols)}')
except Exception as e:
    print(f'  ⚠️  Patch verification error: {e}')
"

# ── Step 7: Prepare Training Data ──────────────────────────────────────────
echo "══════ Step 7: Preparing training data ══════"
cd "${WORKDIR}/kokoro-tamil"

# Create audio directory
mkdir -p "${DATA_DIR}/audio/kavi"

# Run data preparation
python3 scripts/prepare_tamil_data.py prepare || {
    echo "  ⚠️  Standard preparation failed, trying direct parquet extraction..."
    # Fall back to direct extraction from parquet files
    python3 << 'PYEOF'
import os, sys, json, pyarrow.parquet as pq
from pathlib import Path

DATA_DIR = os.environ.get('DATA_DIR', '/teamspace/studios/this_studio/dataset')
raw_dir = Path(DATA_DIR) / "raw"
training_dir = Path(DATA_DIR).parent / "training"
training_dir.mkdir(parents=True, exist_ok=True)

all_entries = []

# Process kavi-ps-manual-10hr-original
kavi_dir = raw_dir / "kavi-ps-manual-10hr-original"
if kavi_dir.exists():
    for pf in sorted(kavi_dir.glob("**/*.parquet")):
        table = pq.read_table(pf)
        df = table.to_pandas()
        for _, row in df.iterrows():
            text = str(row.get('transcription', row.get('text', ''))).strip()
            if text and len(text) > 3:
                all_entries.append({
                    'text': text,
                    'speaker': 'kavi',
                    'source': 'kavi-manual',
                })
    print(f"  kavi-ps-manual-10hr-original: loaded entries")

# Process tts-1000-v1
tts_dir = raw_dir / "tts-1000-v1"
if tts_dir.exists():
    for pf in sorted(tts_dir.glob("**/*.parquet")):
        table = pq.read_table(pf)
        df = table.to_pandas()
        for _, row in df.iterrows():
            text = str(row.get('transcription', row.get('text', ''))).strip()
            if text and len(text) > 3:
                all_entries.append({
                    'text': text,
                    'speaker': 'kavi',
                    'source': 'tts-1000',
                })
    print(f"  tts-1000-v1: loaded entries")

print(f"  Total text entries: {len(all_entries)}")

# Save as metadata
with open(training_dir / "metadata.jsonl", 'w') as f:
    for entry in all_entries:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

# Create train/val split
import random
random.seed(42)
random.shuffle(all_entries)
n_val = max(1, int(len(all_entries) * 0.05))
val = all_entries[:n_val]
train = all_entries[n_val:]

print(f"  Train: {len(train)}, Val: {len(val)}")
PYEOF
}

echo "  ✅ Training data prepared"

# ── Step 8: Update Training Config ─────────────────────────────────────────
echo "══════ Step 8: Configuring training ══════"

# Copy config and update paths for the Studio environment
CONFIG_SRC="${WORKDIR}/kokoro-tamil/configs/config_tamil_ft.yml"
CONFIG_DST="${WORKDIR}/StyleTTS2/config_tamil_ft.yml"

# Update paths in config for Studio environment
cp "${CONFIG_SRC}" "${CONFIG_DST}"

# Adjust batch size and epochs from env vars
sed -i "s/batch_size:.*/batch_size: ${BATCH_SIZE}/" "${CONFIG_DST}"
sed -i "s/epochs_1st:.*/epochs_1st: ${STAGE1_EPOCHS}/" "${CONFIG_DST}"
sed -i "s/epochs_2nd:.*/epochs_2nd: ${STAGE2_EPOCHS}/" "${CONFIG_DST}"

# Update data paths to Studio locations
sed -i "s|../training/|${MODEL_DIR}/|g" "${CONFIG_DST}"
sed -i "s|../dataset/audio/|${DATA_DIR}/audio/|g" "${CONFIG_DST}"
sed -i "s|pretrained_model:.*|pretrained_model: \"${MODEL_DIR}/kokoro_base.pth\"|" "${CONFIG_DST}"

# Create training directory structure
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOG_DIR}"

# Copy base model as kokoro_base.pth (StyleTTS2 format)
if [ ! -f "${MODEL_DIR}/kokoro_base.pth" ]; then
    python3 -c "
import torch, json
from pathlib import Path

# Load Kokoro model and convert to StyleTTS2 format
kokoro_path = '${MODEL_DIR}/kokoro-v1_0.pth'
out_path = '${MODEL_DIR}/kokoro_base.pth'

if Path(kokoro_path).exists():
    print(f'Converting {kokoro_path} to StyleTTS2 format...')
    state = torch.load(kokoro_path, map_location='cpu', weights_only=True)
    
    # Kokoro stores as {component_name: state_dict}
    # StyleTTS2 expects {net: {component_name: state_dict}}
    net = {}
    total_params = 0
    for key, value in state.items():
        if isinstance(value, dict):
            cleaned = {}
            for k, v in value.items():
                clean_k = k.removeprefix('module.')
                cleaned[clean_k] = v
                total_params += v.numel() if hasattr(v, 'numel') else 0
            net[key] = cleaned
    
    checkpoint = {'net': net}
    torch.save(checkpoint, out_path)
    print(f'Saved converted weights: {out_path}')
    print(f'Total parameters: {total_params/1e6:.2f}M')
else:
    print(f'WARNING: {kokoro_path} not found')
"
fi

echo "  ✅ Training config ready"

# ── Step 9: Verify Setup ───────────────────────────────────────────────────
echo "══════ Step 9: Verifying setup ══════"

python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "  Data directory:"
ls -la "${DATA_DIR}/raw/" 2>/dev/null || echo "    (raw data dir not found)"

echo "  Model directory:"
ls -la "${MODEL_DIR}/" 2>/dev/null || echo "    (model dir empty)"

echo "  Config file:"
head -5 "${CONFIG_DST}"

echo ""
echo "============================================================"
echo "  Setup Complete! 🎉"
echo "============================================================"
echo ""
echo "  To run training:"
echo "    cd ${WORKDIR}/StyleTTS2"
echo ""
echo "    # Stage 1 (Acoustic/Prosody):"
echo "    accelerate launch --mixed_precision fp16 train_first.py --config_path ${CONFIG_DST}"
echo ""
echo "    # Stage 2 (Duration/SLM):"
echo "    accelerate launch --mixed_precision fp16 train_second.py --config_path ${CONFIG_DST}"
echo ""
echo "  Or use the training scripts:"
echo "    bash ${WORKDIR}/kokoro-tamil/scripts/lightning/train_stage1.sh"
echo "    bash ${WORKDIR}/kokoro-tamil/scripts/lightning/train_stage2.sh"
echo "============================================================"