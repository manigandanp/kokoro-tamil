#!/usr/bin/env python3
"""
Kokoro Tamil Fine-Tuning — Modal.com Pipeline
===============================================

Complete 2-stage Kokoro TTS fine-tuning on Modal GPUs.

Uses MODAL_FALLBACK_TOKEN_ID / MODAL_FALLBACK_TOKEN_SECRET env vars.

Key design decisions:
- Clones semidark/kokoro-deutsch (includes kokoro + StyleTTS2 submodules)
- Copies kokoro-tamil scripts over the top (config, G2P, etc.)
- Downloads datasets to persistent volume
- Converts Kokoro base weights to StyleTTS2 format
- Runs Stage 1 / Stage 2 training on A100-80GB

Usage:
    # Run full pipeline (download + prepare + Stage 1 + Stage 2)
    MODAL_TOKEN_ID=$MODAL_FALLBACK_TOKEN_ID MODAL_TOKEN_SECRET=$MODAL_FALLBACK_TOKEN_SECRET \
        modal run scripts/modal_train.py --hf-token YOUR_HF_TOKEN

    # Run only Stage 1
    modal run scripts/modal_train.py --stage 1 --hf-token YOUR_HF_TOKEN

    # Skip download (use existing volume data)
    modal run scripts/modal_train.py --stage 1 --skip-download --hf-token YOUR_HF_TOKEN

    # Test inference after training
    modal run scripts/modal_train.py --test-only --hf-token YOUR_HF_TOKEN
"""

import modal
import os
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Modal App & Volume
# ─────────────────────────────────────────────────────────────────────────────

app = modal.App("kokoro-tamil-training")

# Persistent volume for datasets, model weights, checkpoints
vol = modal.Volume.from_name("kokoro-tamil-vol", create_if_missing=True)

DATA_DIR = "/root/data"
STYTTS2_DIR = f"{DATA_DIR}/StyleTTS2"
KOKORO_DEUTSCH_DIR = f"{DATA_DIR}/kokoro-deutsch"
KOKORO_TAMIL_DIR = f"{DATA_DIR}/kokoro-tamil"
TRAINING_DIR = f"{DATA_DIR}/training"
DATASET_DIR = f"{DATA_DIR}/dataset"
LOGS_DIR = f"{STYTTS2_DIR}/logs/kokoro-tamil"

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for patching StyleTTS2 code
# ─────────────────────────────────────────────────────────────────────────────

def patch_models_py(models_path):
    """
    Patch models.py to fix fragile patching issues and Conv2d kernel size errors.
    
    This function:
    1. Removes ALL previous broken patches (inline comments, orphaned imports, etc.)
    2. Applies the Conv2d fix: padding=0 → padding=2 to prevent kernel size errors
    3. Returns True if patches were applied, False if file doesn't exist
    """
    import re as _re
    
    if not os.path.exists(models_path):
        return False
    
    with open(models_path, "r") as f:
        models_src = f.read()
    
    patched = False
    
    # Remove ALL previous broken patches comprehensively
    
    # 1. Remove any inline comments inside spectral_norm() calls that break Python syntax
    broken_conv5 = "nn.Conv2d(dim_out, dim_out, 5, 1, 2)  # padding=2 (same) to prevent kernel size error"
    if broken_conv5 in models_src:
        models_src = models_src.replace(broken_conv5, "nn.Conv2d(dim_out, dim_out, 5, 1, 2)")
        patched = True
        print("✓ Removed broken inline comment from previous Conv2d patch")
    
    # 2. Remove ALL instances of orphaned import lines (not just when paired)
    models_src = _re.sub(r'\s*import torch\.nn\.functional as _F\n', '', models_src)
    models_src = _re.sub(r'\s*_min_dim = \d+\n', '', models_src)
    
    # 3. Remove ALL "Safety: pad small inputs" comment blocks AND the code between them
    if "Safety: pad small inputs" in models_src:
        print("⚠ Found old runtime-padding patches, removing them (superseded by Conv2d padding=2)")
        # Remove all padding blocks with comprehensive regex
        models_src = _re.sub(
            r'\n\s*# Safety: pad small inputs.*?\n\s*x = _F\.pad\(x, \([^)]+\)[^)]*\)\n',
            '\n', models_src, flags=_re.DOTALL
        )
        patched = True
    
    # 4. Apply the Conv2d fix: padding=0 → padding=2
    old_conv5 = "nn.Conv2d(dim_out, dim_out, 5, 1, 0)"
    new_conv5 = "nn.Conv2d(dim_out, dim_out, 5, 1, 2)"
    if old_conv5 in models_src:
        models_src = models_src.replace(old_conv5, new_conv5)
        conv_count = models_src.count(new_conv5)
        print(f"✓ Patched models.py: Conv2d(5,1,0)→Conv2d(5,1,2) in {conv_count} places")
        patched = True
    elif new_conv5 in models_src:
        print("✓ Conv2d padding fix already applied in models.py")
    else:
        print("⚠ Could not find Conv2d(dim_out, dim_out, 5, 1, 0) in models.py")
    
    # Write back the cleaned version
    if patched:
        with open(models_path, "w") as f:
            f.write(models_src)
    
    return patched


def patch_slmadv_py(slmadv_path):
    """
    Patch slmadv.py to guard against too-small mel segments.
    
    Ensures mel_len >= 40 in SLMAdversarialLoss.forward() to prevent
    Conv2d kernel size errors after downsampling.
    
    Returns True if patches were applied, False if file doesn't exist.
    """
    if not os.path.exists(slmadv_path):
        return False
    
    with open(slmadv_path, "r") as f:
        slmadv_src = f.read()
    
    # The original line: mel_len = max(int(min(output_lengths) / 2 - 1), self.min_len // 2)
    # Change the floor from self.min_len // 2 to max(self.min_len // 2, 40)
    old_mellen = "mel_len = max(int(min(output_lengths) / 2 - 1), self.min_len // 2)"
    new_mellen = "mel_len = max(int(min(output_lengths) / 2 - 1), max(self.min_len // 2, 40))"
    
    if old_mellen in slmadv_src and "max(self.min_len // 2, 40)" not in slmadv_src:
        slmadv_src = slmadv_src.replace(old_mellen, new_mellen)
        with open(slmadv_path, "w") as f:
            f.write(slmadv_src)
        print("✓ Patched slmadv.py: mel_len floor increased to max(min_len//2, 40)")
        return True
    elif "max(self.min_len // 2, 40)" in slmadv_src:
        print("✓ slmadv.py already patched with mel_len floor guard")
        return False
    else:
        print("⚠ Could not find mel_len line in slmadv.py to patch")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Build Image
# ─────────────────────────────────────────────────────────────────────────────

# Key pitfalls (from Lightning AI experience):
# 1. transformers>=5.7 blocks torch<2.6 — must pin transformers==4.47.1
# 2. torch/torchvision/torchaudio version mismatch — pin to 2.5.1+cu124
# 3. Must build monotonic_align for StyleTTS2 (Cython extension)
# 4. Must symlink wav/ dir inside StyleTTS2
# 5. espeak-ng Tamil is poor — custom G2P in prepare_tamil_data.py
# 6. semidark/StyleTTS2 has kokoro_symbols.py (required for training)
# 7. semidark/StyleTTS2 has kokoro_tb_utils.py (German — we override)
# 8. semidark/StyleTTS2 includes pretrained Utils/ (JDC, ASR, PLBERT)
# 9. accelerate is used for training (single GPU mode)

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "python3-dev", "python3-pip", "git", "git-lfs", "espeak-ng",
        "libsndfile1-dev", "libespeak-ng-dev", "cmake", "build-essential",
        "wget", "curl", "ffmpeg", "clang",
    )
    .run_commands(
        # Install PyTorch with CUDA 12.4 support FIRST (before any other ML packages)
        "pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 "
        "--index-url https://download.pytorch.org/whl/cu124",
        # Git LFS for model downloads
        "git lfs install",
    )
    .env({"_REBUILD_TRIGGER": "2026-05-02-v11"})  # Force image rebuild
    .pip_install(
        # Transformers — MUST pin to 4.47.1 to avoid torch>=2.6 requirement
        "transformers==4.47.1",
        # Accelerate for distributed/fp16 training
        "accelerate>=0.33.0",
        # Data & audio
        "datasets>=3.0.0", "huggingface_hub>=0.25.0",
        "soundfile", "librosa", "scipy", "pyarrow",
        # Training utilities
        "pyyaml", "tensorboard", "munch", "phonemizer", "Cython",
        "einops", "einops-exts", "click", "sentencepiece",
        "rotary_embedding_torch", "misaki[espeak]", "ninja", "matplotlib",
    )
    .add_local_dir(
        "/home/ubuntu/kokoro-tamil",
        remote_path="/opt/kokoro-tamil",
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download datasets & base model
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    timeout=7200,     # 2 hours
    memory=32768,     # 32 GB
    cpu=4,
)
def download_data(hf_token: str):
    """Download all Tamil TTS datasets and Kokoro base model from HuggingFace."""
    import os
    from huggingface_hub import login, snapshot_download, hf_hub_download

    login(token=hf_token)
    os.environ["HF_TOKEN"] = hf_token

    # ── Download datasets ────────────────────────────────────────────────
    datasets = {
        "kavi-ps-manual-10hr-original": {
            "repo": "mastermani305/kavi-ps-manual-10hr-original",
            "private": True,
        },
        "tts-1000-v1": {
            "repo": "mastermani305/tts-1000-v1",
            "private": False,
        },
    }

    raw_dir = f"{DATASET_DIR}/raw"
    for name, info in datasets.items():
        out_dir = f"{raw_dir}/{name}"
        if os.path.exists(out_dir) and os.listdir(out_dir):
            print(f"✓ SKIP (exists): {name}")
            continue
        print(f"⬇ Downloading: {name}...")
        snapshot_download(
            info["repo"],
            repo_type="dataset",
            token=hf_token,
            local_dir=out_dir,
        )
        print(f"✓ Done: {name}")

    # ── Download Kokoro base model ────────────────────────────────────────
    model_dir = f"{DATA_DIR}/base_model"
    os.makedirs(model_dir, exist_ok=True)

    for filename in ["kokoro-v1_0.pth"]:
        out_path = f"{model_dir}/{filename}"
        if os.path.exists(out_path):
            print(f"✓ SKIP (exists): {filename}")
            continue
        print(f"⬇ Downloading model: {filename}...")
        hf_hub_download("hexgrad/Kokoro-82M", filename, local_dir=model_dir, token=hf_token)
        print(f"✓ Done: {filename}")

    vol.commit()
    print("✅ Data download complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Prepare training data
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    timeout=3600,     # 60 min (data processing takes time)
    memory=32768,     # 32 GB (need RAM for parquet processing)
    cpu=4,
)
def prepare_data():
    """Process downloaded datasets: extract WAVs, phonemize, generate train/val lists."""
    import json
    import os
    import re
    import random
    import sys
    import io
    import shutil
    import torch
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    from huggingface_hub import hf_hub_download

    # ── Ensure dirs ─────────────────────────────────────────────────────────
    audio_dir = f"{DATASET_DIR}/audio/kavi"
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)

    # ── Copy kokoro-tamil scripts from embedded mount (always refresh) ────
    os.makedirs(KOKORO_TAMIL_DIR, exist_ok=True)
    src_mount = "/opt/kokoro-tamil/scripts"
    dst_scripts = f"{KOKORO_TAMIL_DIR}/scripts"
    if os.path.exists(src_mount):
        os.makedirs(dst_scripts, exist_ok=True)
        for fname in os.listdir(src_mount):
            src_file = os.path.join(src_mount, fname)
            dst_file = os.path.join(dst_scripts, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
        print("✓ Copied kokoro-tamil scripts from mount")
    else:
        if not os.path.exists(KOKORO_TAMIL_DIR):
            shutil.copytree("/opt/kokoro-tamil", KOKORO_TAMIL_DIR)

    # ── Add scripts to path so we can import prepare_tamil_data ─────────────
    sys.path.insert(0, f"{KOKORO_TAMIL_DIR}/scripts")
    from prepare_tamil_data import tamil_to_ipa, phonemize_tamil, TAMIL_OOD_SENTENCES

    # ── Helper: Load parquet dataset ────────────────────────────────────────
    def load_parquet_dataset(dataset_dir):
        """Load entries from HF parquet dataset directory."""
        import pyarrow.parquet as pq
        entries = []
        for pf in sorted(Path(dataset_dir).glob("**/*.parquet")):
            table = pq.read_table(pf)
            df = table.to_pandas()
            for _, row in df.iterrows():
                entries.append(dict(row))
        return entries

    # ── Process each dataset ────────────────────────────────────────────────
    all_entries = []

    # 1. kavi-ps-manual-10hr-original
    kavi_dir = f"{DATASET_DIR}/raw/kavi-ps-manual-10hr-original"
    if os.path.exists(kavi_dir):
        print(f"\nProcessing kavi-ps-manual-10hr-original...")
        entries = load_parquet_dataset(kavi_dir)
        print(f"  Loaded {len(entries)} raw rows")
        for row in entries:
            text = row.get('transcription', '')
            if not text or len(text) < 3:
                continue
            ipa = tamil_to_ipa(text)
            if len(ipa) < 5:
                continue
            fname = row.get('file_name', f"kavi_{len(all_entries)}")
            audio = row.get('audio', {})
            if isinstance(audio, dict) and 'bytes' in audio:
                audio_bytes = audio['bytes']
            else:
                audio_bytes = None
            all_entries.append({
                'source': 'kavi',
                'file_name': fname,
                'text': text,
                'ipa': ipa,
                'speaker': 'kavi',
                'audio_bytes': audio_bytes,
            })
        print(f"  Valid entries: {len([e for e in all_entries if e['source']=='kavi'])}")
    else:
        print(f"⚠ Dataset dir not found: {kavi_dir}")

    # 2. tts-1000-v1 (public, supplementary)
    tts1k_dir = f"{DATASET_DIR}/raw/tts-1000-v1"
    if os.path.exists(tts1k_dir):
        print(f"\nProcessing tts-1000-v1...")
        entries = load_parquet_dataset(tts1k_dir)
        print(f"  Loaded {len(entries)} raw rows")
        count_before = len(all_entries)
        for row in entries:
            text = row.get('transcription', '')
            if not text or len(text) < 3:
                continue
            ipa = tamil_to_ipa(text)
            if len(ipa) < 5:
                continue
            fname = row.get('file_name', f"tts1k_{len(all_entries)}")
            audio = row.get('audio', {})
            if isinstance(audio, dict) and 'bytes' in audio:
                audio_bytes = audio['bytes']
            else:
                audio_bytes = None
            all_entries.append({
                'source': 'tts-1k',
                'file_name': fname,
                'text': text,
                'ipa': ipa,
                'speaker': 'kavi',
                'audio_bytes': audio_bytes,
            })
        print(f"  Valid entries: {len(all_entries) - count_before}")
    else:
        print(f"⚠ Dataset dir not found: {tts1k_dir}")

    if not all_entries:
        # List what we have for debugging
        print(f"\n⚠ No entries found! Listing {DATASET_DIR}/raw/:")
        raw_dir = f"{DATASET_DIR}/raw"
        if os.path.exists(raw_dir):
            for d in os.listdir(raw_dir):
                print(f"  {d}/")
                sub = os.path.join(raw_dir, d)
                if os.path.isdir(sub):
                    for item in os.listdir(sub)[:10]:
                        print(f"    {item}")
        raise RuntimeError("No training entries found. Check dataset downloads.")

    # ── Deduplicate ──────────────────────────────────────────────────────────
    seen = set()
    unique = []
    for e in all_entries:
        if e['text'] not in seen:
            seen.add(e['text'])
            unique.append(e)
    print(f"\nTotal: {len(all_entries)}, Unique: {len(unique)}")

    # ── Train/val split ────────────────────────────────────────────────────
    rng = random.Random(42)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * 0.05))
    val_entries = unique[:n_val]
    train_entries = unique[n_val:]
    print(f"Split: {len(train_entries)} train / {len(val_entries)} val")

    # ── Extract WAV audio ──────────────────────────────────────────────────
    print("\nExtracting WAV audio files...")
    extracted = 0
    errors = 0
    for entry in unique:
        audio_bytes = entry.get('audio_bytes')
        if not audio_bytes:
            errors += 1
            continue
        fname = entry['file_name']
        if not fname.endswith('.wav'):
            fname = fname.rsplit('.', 1)[0] + '.wav' if '.' in fname else fname + '.wav'
        speaker_subdir = entry.get('speaker', 'kavi')
        out_path = os.path.join(f"{DATASET_DIR}/audio", speaker_subdir, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        wav_rel_path = f"{speaker_subdir}/{fname}"
        if os.path.exists(out_path):
            extracted += 1
            entry['wav_path'] = wav_rel_path
            continue
        try:
            waveform, sr = sf.read(io.BytesIO(audio_bytes))
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            if sr != 24000:
                from scipy import signal as sig
                num = int(len(waveform) * 24000 / sr)
                waveform = sig.resample(waveform, num)
                sr = 24000
            sf.write(out_path, waveform, 24000, subtype='PCM_16')
            extracted += 1
            entry['wav_path'] = wav_rel_path
        except Exception as ex:
            errors += 1
            if errors <= 5:
                print(f"  Error extracting {fname}: {ex}")
    print(f"Extracted: {extracted}, Errors: {errors}")

    # ── Build speaker ID map (string → int) ──────────────────────────────
    unique_speakers = sorted(set(e['speaker'] for e in train_entries + val_entries if isinstance(e['speaker'], str)))
    speaker_map = {name: idx for idx, name in enumerate(unique_speakers)}
    print(f"Speaker map: {speaker_map}")

    # ── Write training lists ────────────────────────────────────────────────
    # StyleTTS2 format: wav_path|ipa_phonemes|speaker_id
    # NOTE: speaker_id MUST be an integer (meldataset.py calls int() on it)
    # NOTE: wav_path is relative to root_path (our root_path = ../dataset/audio)
    def write_list(entries, path):
        with open(path, 'w') as f:
            for e in entries:
                wp = e.get('wav_path', e.get('file_name', ''))
                if not wp.endswith('.wav'):
                    wp = wp.rsplit('.', 1)[0] + '.wav' if '.' in wp else wp + '.wav'
                # Map string speaker names to integer IDs
                spk = e['speaker']
                if isinstance(spk, str):
                    spk = int(speaker_map.get(spk, 0))
                f.write(f"{wp}|{e['ipa']}|{spk}\n")

    write_list(train_entries, f"{TRAINING_DIR}/train_list.txt")
    write_list(val_entries, f"{TRAINING_DIR}/val_list.txt")
    print(f"Wrote {TRAINING_DIR}/train_list.txt ({len(train_entries)} lines)")
    print(f"Wrote {TRAINING_DIR}/val_list.txt ({len(val_entries)} lines)")

    # ── Write OOD texts ────────────────────────────────────────────────────
    ood_phonemes = [tamil_to_ipa(t) for t in TAMIL_OOD_SENTENCES]
    with open(f"{TRAINING_DIR}/OOD_texts.txt", 'w') as f:
        f.write('\n'.join(ood_phonemes) + '\n')
    print(f"Wrote OOD_texts.txt ({len(ood_phonemes)} sentences)")

    # ── Convert Kokoro base weights ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Converting Kokoro base weights...")
    print("=" * 60)
    model_path = f"{DATA_DIR}/base_model/kokoro-v1_0.pth"
    output_path = f"{TRAINING_DIR}/kokoro_base.pth"
    if os.path.exists(model_path) and not os.path.exists(output_path):
        kokoro_state = torch.load(model_path, map_location="cpu", weights_only=True)
        # Kokoro weights are organized by component (e.g., 'decoder', 'text_encoder', etc.)
        # train_first.py's load_checkpoint expects state["net"][component_key] = {param: tensor}
        # Remove 'module.' prefix from DDP training if present
        net = {}
        for component, state_dict in kokoro_state.items():
            cleaned = {}
            for key, tensor in state_dict.items():
                clean_key = key.removeprefix("module.")
                cleaned[clean_key] = tensor
            net[component] = cleaned
            print(f"  {component}: {len(cleaned)} tensors")
        checkpoint = {"net": net}
        torch.save(checkpoint, output_path)
        print(f"✓ Saved converted weights: {output_path}")
    elif os.path.exists(output_path):
        print(f"✓ Skip (already exists): {output_path}")
    else:
        print(f"⚠ Missing base model: {model_path}")

    vol.commit()
    print("✅ Data preparation complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Setup StyleTTS2 + patches
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    timeout=600,      # 10 min
    memory=16384,    # 16 GB
    cpu=4,
)
def setup_training(hf_token: str):
    """Clone kokoro-deutsch (with submodules), apply Tamil patches, build monotonic_align."""
    import os
    import subprocess
    import shutil

    os.environ["HF_TOKEN"] = hf_token

    # ── Clone kokoro-deutsch (includes StyleTTS2 + kokoro submodules) ──────
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(DATA_DIR)

    if not os.path.exists(KOKORO_DEUTSCH_DIR):
        print("⬇ Cloning kokoro-deutsch (--recursive)...")
        result = subprocess.run(
            ["git", "clone", "--recursive",
             "https://github.com/semidark/kokoro-deutsch.git", KOKORO_DEUTSCH_DIR],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"Clone failed: {result.stderr[-1000:]}")
            raise RuntimeError(f"Failed to clone kokoro-deutsch: {result.returncode}")
        print("✓ Cloned kokoro-deutsch with submodules")
    else:
        print("✓ kokoro-deutsch exists")

    # ── Ensure submodules are initialized ──────────────────────────────────
    if not os.path.exists(f"{KOKORO_DEUTSCH_DIR}/StyleTTS2/train_first.py"):
        print("⬇ Initializing submodules...")
        result = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            capture_output=True, text=True,
            cwd=KOKORO_DEUTSCH_DIR,
        )
        if result.returncode != 0:
            print(f"Submodule init failed: {result.stderr[-1000:]}")
            raise RuntimeError(f"Failed to init submodules: {result.returncode}")
        print("✓ Submodules initialized")

    # ── Create symlink: StyleTTS2 → kokoro-deutsch/StyleTTS2 ──────────────
    # This makes our paths consistent: /root/data/StyleTTS2 → /root/data/kokoro-deutsch/StyleTTS2
    if os.path.islink(STYTTS2_DIR):
        # Existing symlink — just verify it points to the right target
        current_target = os.readlink(STYTTS2_DIR)
        if current_target != f"{KOKORO_DEUTSCH_DIR}/StyleTTS2":
            print(f"⚠ Updating StyleTTS2 symlink: {STYTTS2_DIR} → {current_target} (was pointing to wrong target)")
            os.unlink(STYTTS2_DIR)
        else:
            print(f"✓ StyleTTS2 symlink already correct: {STYTTS2_DIR} → {current_target}")
    elif os.path.isdir(STYTTS2_DIR):
        # Real directory (e.g. old upstream clone) — only remove if KOKORO_DEUTSCH_DIR/StyleTTS2 exists
        if os.path.exists(f"{KOKORO_DEUTSCH_DIR}/StyleTTS2") and not os.path.samefile(STYTTS2_DIR, f"{KOKORO_DEUTSCH_DIR}/StyleTTS2"):
            # Different directory — back up and replace with symlink
            print(f"⚠ Removing old StyleTTS2 directory to replace with symlink...")
            backup_dir = f"{STYTTS2_DIR}.old_upstream"
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            os.rename(STYTTS2_DIR, backup_dir)
            print(f"  Backed up to {backup_dir}")
        elif not os.path.exists(f"{KOKORO_DEUTSCH_DIR}/StyleTTS2"):
            # Keep the existing directory since we need it
            print(f"✓ Using existing StyleTTS2 directory (submodule not initialized)")

    if not os.path.exists(STYTTS2_DIR):
        os.symlink(f"{KOKORO_DEUTSCH_DIR}/StyleTTS2", STYTTS2_DIR)
        print(f"✓ Symlinked: {STYTTS2_DIR} → {KOKORO_DEUTSCH_DIR}/StyleTTS2")

    # ── Copy kokoro-tamil scripts from local mount (always refresh) ─────────
    os.makedirs(KOKORO_TAMIL_DIR, exist_ok=True)
    src_mount = "/opt/kokoro-tamil/scripts"
    dst_scripts = f"{KOKORO_TAMIL_DIR}/scripts"
    if os.path.exists(src_mount):
        os.makedirs(dst_scripts, exist_ok=True)
        for fname in os.listdir(src_mount):
            src_file = os.path.join(src_mount, fname)
            dst_file = os.path.join(dst_scripts, fname)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, dst_file)
        print(f"✓ Copied kokoro-tamil scripts from mount")
    else:
        print(f"⚠ Mount /opt/kokoro-tamil/scripts not found, using volume copy")
        if not os.path.exists(f"{KOKORO_TAMIL_DIR}/scripts"):
            shutil.copytree("/opt/kokoro-tamil", KOKORO_TAMIL_DIR)

    # ── Symlink training & dataset dirs inside kokoro-deutsch ────────────────
    # CRITICAL: StyleTTS2 is symlinked /root/data/StyleTTS2 → /root/data/kokoro-deutsch/StyleTTS2
    # Python's os.getcwd() and os.path.realpath() resolve through symlinks to the
    # REAL path (/root/data/kokoro-deutsch/StyleTTS2).  Any relative path like
    # ../training/ or ../dataset/ in the config (or in train_first.py) therefore
    # resolves inside /root/data/kokoro-deutsch/ rather than /root/data/.
    #
    # Even though our config_tamil_ft.yml uses ABSOLUTE paths for data_params,
    # train_first.py may resolve some paths relative to CWD or to the config
    # file's real directory.  We MUST ensure that /root/data/kokoro-deutsch/{training,dataset}
    # point to the real data directories.
    #
    # NOTE: kokoro-deutsch/training/ is a non-empty directory from git (contains
    # German OOD_texts.txt, config.json, kokoro_symbols.py).  We must forcefully
    # replace it with a symlink; the old content is backed up.
    for dirname in ["training", "dataset"]:
        src = f"{DATA_DIR}/{dirname}"
        dst = f"{KOKORO_DEUTSCH_DIR}/{dirname}"

        if os.path.islink(dst):
            # Existing symlink — remove and recreate
            os.unlink(dst)
        elif os.path.isdir(dst):
            # Real directory (e.g. from git clone) — back up and replace with symlink
            backup = f"{dst}.gitbackup"
            if os.path.exists(backup):
                shutil.rmtree(backup)
            os.rename(dst, backup)
            print(f"  Backed up existing {dst} → {backup}")
        elif os.path.exists(dst):
            # Regular file — just remove it
            os.remove(dst)

        os.symlink(src, dst)
        print(f"✓ Symlinked: {dst} → {src}")

    # ── CRITICAL: Patch kokoro_tb_utils.py with Tamil version ───────────────
    # The semidark/StyleTTS2 version uses German G2P — we replace with Tamil
    print("🔧 Patching kokoro_tb_utils.py with Tamil version...")
    # Try image-baked path first (always fresh), then fall back to volume copy
    image_tb_utils = "/opt/kokoro-tamil/scripts/kokoro_tb_utils_tamil.py"
    volume_tb_utils = f"{KOKORO_TAMIL_DIR}/scripts/kokoro_tb_utils_tamil.py"
    src_tb_utils = image_tb_utils if os.path.isfile(image_tb_utils) else volume_tb_utils
    if not os.path.isfile(src_tb_utils):
        raise FileNotFoundError(
            f"kokoro_tb_utils_tamil.py not found at {image_tb_utils} or {volume_tb_utils}. "
            f"Ensure the kokoro-tamil scripts are mounted via add_local_dir or present on the volume."
        )
    dst_tb_utils = f"{STYTTS2_DIR}/kokoro_tb_utils.py"
    shutil.copy2(src_tb_utils, dst_tb_utils)
    print(f"✓ Copied {src_tb_utils} -> {dst_tb_utils}")

    # ── Build monotonic_align ─────────────────────────────────────────────
    # semidark/StyleTTS2 has monotonic_align as a git submodule
    ma_dir = f"{STYTTS2_DIR}/monotonic_align"
    if os.path.isfile(f"{ma_dir}/setup.py"):
        os.chdir(ma_dir)
        print("Building monotonic_align from source...")
        result = subprocess.run(
            ["python3", "setup.py", "build_ext", "--inplace"],
            capture_output=True, text=True,
        )
        if result.stdout:
            print(result.stdout[-500:])
        if result.returncode != 0:
            print("Build output:", result.stderr[-1000:])
            # Fallback: pip install from resemble-ai
            print("⚠ Build failed, installing monotonic_align via pip...")
            result2 = subprocess.run(
                ["pip", "install", "git+https://github.com/resemble-ai/monotonic_align.git"],
                capture_output=True, text=True,
            )
            if result2.returncode != 0:
                raise RuntimeError(f"Failed to install monotonic_align: {result2.stderr[-500:]}")
            print("✓ monotonic_align installed via pip")
        else:
            # Install as editable package so it's importable from anywhere
            os.chdir(STYTTS2_DIR)
            subprocess.run(
                ["pip", "install", "-e", ma_dir],
                capture_output=True, text=True,
            )
            print("✓ monotonic_align built from source")
    else:
        # No setup.py — pip install from GitHub
        print("Installing monotonic_align via pip...")
        result = subprocess.run(
            ["pip", "install", "git+https://github.com/resemble-ai/monotonic_align.git"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install monotonic_align: {result.stderr[-500:]}")
        print("✓ monotonic_align installed via pip")

    # ── Symlink WAV directory ──────────────────────────────────────────────
    # StyleTTS2 convention: StyleTTS2/wav/ -> audio files
    wav_link = f"{STYTTS2_DIR}/wav"
    audio_dir = f"{DATASET_DIR}/audio/kavi"
    if os.path.islink(wav_link):
        os.unlink(wav_link)
    elif os.path.isdir(wav_link):
        shutil.rmtree(wav_link)
    elif os.path.exists(wav_link):
        os.remove(wav_link)

    os.symlink(audio_dir, wav_link)
    print(f"✓ Symlinked: {wav_link} → {audio_dir}")

    # ── Copy Tamil training config ──────────────────────────────────────────
    # Try image-baked path first (always fresh), then fall back to volume copy
    image_config = "/opt/kokoro-tamil/configs/config_tamil_ft.yml"
    volume_config = f"{KOKORO_TAMIL_DIR}/configs/config_tamil_ft.yml"
    config_src = image_config if os.path.isfile(image_config) else volume_config
    config_dst = f"{STYTTS2_DIR}/config_tamil_ft.yml"
    shutil.copy2(config_src, config_dst)
    print(f"✓ Copied config: {config_dst}")

    # Also copy to StyleTTS2/Configs/ (default search path)
    config_dst2 = f"{STYTTS2_DIR}/Configs/config_tamil_ft.yml"
    os.makedirs(f"{STYTTS2_DIR}/Configs", exist_ok=True)
    shutil.copy2(config_src, config_dst2)
    print(f"✓ Copied config: {config_dst2}")

    # ── Patch models.py and slmadv.py: Fix issues BEFORE import check ──────
    # Apply full patching logic (not just limited pre-fix) to ensure
    # import verification sees clean Python syntax
    models_path = f"{STYTTS2_DIR}/models.py"
    slmadv_path = f"{STYTTS2_DIR}/Modules/slmadv.py"
    
    if os.path.exists(models_path):
        patch_models_py(models_path)
    
    if os.path.exists(slmadv_path):
        patch_slmadv_py(slmadv_path)

    # ── Verify imports ────────────────────────────────────────────────────
    os.chdir(STYTTS2_DIR)
    result = subprocess.run(
        ["python3", "-c",
         "from models import *; from transformers import AlbertModel; "
         "import monotonic_align; from kokoro_symbols import TextCleaner; "
         "print('✓ All imports OK')"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Import error:", result.stderr[-2000:])
        raise RuntimeError("Import verification failed")

    # ── Verify training lists ─────────────────────────────────────────────
    for name in ["train_list.txt", "val_list.txt", "OOD_texts.txt"]:
        path = f"{TRAINING_DIR}/{name}"
        if os.path.exists(path):
            with open(path) as f:
                count = sum(1 for _ in f)
            print(f"✓ {name}: {count} lines")
            # Show first line for path verification
            with open(path) as f:
                first_line = f.readline().strip()
            print(f"  First line: {first_line[:80]}...")
        else:
            print(f"⚠ Missing: {path}")

    # ── Verify base weights ────────────────────────────────────────────────
    base_weights = f"{TRAINING_DIR}/kokoro_base.pth"
    if os.path.exists(base_weights):
        check_cmd = (
            "import torch; "
            f"s=torch.load('{base_weights}', map_location='cpu', weights_only=True); "
            "print('✓ Base weights loaded, keys:', list(s.keys())); "
            "net=s['net']; print('Components:', list(net.keys())); "
            "print('Decoder keys:', len(net.get('decoder', dict())))"
        )
        result = subprocess.run(
            ["python3", "-c", check_cmd],
            capture_output=True, text=True,
        )
        print(result.stdout.strip())
        if result.returncode != 0:
            print("⚠ Weight check error:", result.stderr[-500:])
    else:
        print(f"⚠ Missing base weights: {base_weights}")

    # ── Verify pretrained models (JDC, ASR, PLBERT) ────────────────────────
    for model_name, model_path in [
        ("JDC (F0)", f"{STYTTS2_DIR}/Utils/JDC/bst.t7"),
        ("ASR", f"{STYTTS2_DIR}/Utils/ASR/epoch_00080.pth"),
        ("ASR config", f"{STYTTS2_DIR}/Utils/ASR/config.yml"),
        ("PLBERT", f"{STYTTS2_DIR}/Utils/PLBERT/step_1000000.t7"),
    ]:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ {model_name}: {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"⚠ Missing {model_name}: {model_path}")

    vol.commit()
    print("✅ Training setup complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Stage 1 Training (Acoustic/Prosody model)
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=train_image,
    gpu="A100-80GB",
    volumes={DATA_DIR: vol},
    timeout=86400,     # 24 hours max
    memory=65536,      # 64 GB
    cpu=8,
)
def train_stage1(hf_token: str, resume_epoch: int = 0):
    """Fine-tune Kokoro Stage 1 (acoustic/prosody model) on A100-80GB."""
    import torch
    import os
    import shutil
    import subprocess

    os.environ["HF_TOKEN"] = hf_token

    print(f"🔧 Stage 1 Training")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Resume from epoch: {resume_epoch}")

    # ── Ensure essential symlinks exist ───────────────────────────────────
    # These must exist so that path resolution works whether train_first.py
    # uses absolute paths from config OR resolves ../training, ../dataset
    # relative to the REAL StyleTTS2 path (through the symlink).
    for dirname in ["training", "dataset"]:
        src = f"{DATA_DIR}/{dirname}"
        dst = f"{KOKORO_DEUTSCH_DIR}/{dirname}"
        if os.path.islink(dst):
            pass  # symlink exists — OK
        elif os.path.isdir(dst):
            # Real directory from git — need to back up and replace with symlink
            backup = f"{dst}.gitbackup"
            if not os.path.exists(backup):
                os.rename(dst, backup)
                print(f"  Backed up {dst} → {backup}")
            else:
                shutil.rmtree(dst)  # remove the gitbackup we already renamed
                print(f"  Removed existing {dst}")
            os.symlink(src, dst)
            print(f"✓ Created symlink: {dst} → {src}")
        elif not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"✓ Created symlink: {dst} → {src}")

    # Ensure wav symlink exists and points to correct target
    wav_link = f"{STYTTS2_DIR}/wav"
    audio_target = f"{DATASET_DIR}/audio/kavi"
    if os.path.islink(wav_link):
        if os.readlink(wav_link) != audio_target:
            os.unlink(wav_link)
            os.symlink(audio_target, wav_link)
    elif os.path.isdir(wav_link):
        shutil.rmtree(wav_link)
        os.symlink(audio_target, wav_link)
    elif not os.path.exists(wav_link):
        os.symlink(audio_target, wav_link)

    # Build monotonic_align (if needed — setup_training should have installed it)
    os.chdir(STYTTS2_DIR)
    try:
        import monotonic_align
        print("✓ monotonic_align already available")
    except ImportError:
        print("⚠ monotonic_align not found, installing via pip...")
        subprocess.run(
            ["pip", "install", "git+https://github.com/resemble-ai/monotonic_align.git"],
            capture_output=True, text=True, check=True,
        )

    # ── Diagnostics: verify data paths ─────────────────────────────────────
    train_list = f"{TRAINING_DIR}/train_list.txt"
    if os.path.exists(train_list):
        with open(train_list) as f:
            first_line = f.readline().strip()
        print(f"📋 First training entry: {first_line[:80]}")
        # Verify the audio file exists
        parts = first_line.split('|')
        if len(parts) >= 1:
            wav_rel = parts[0]
            # root_path from config = /root/data/dataset/audio (absolute)
            full_path = os.path.join(f"{DATASET_DIR}/audio", wav_rel)
            print(f"  Audio file: {full_path} → exists={os.path.exists(full_path)}")

    # Also verify path resolution through kokoro-deutsch symlink
    real_sty_dir = os.path.realpath(STYTTS2_DIR)
    print(f"  StyleTTS2 symlink: {STYTTS2_DIR} → {real_sty_dir}")
    for dirname in ["training", "dataset"]:
        dst = f"{KOKORO_DEUTSCH_DIR}/{dirname}"
        if os.path.islink(dst):
            target = os.readlink(dst)
            valid = os.path.exists(dst)
            check_file = os.path.join(dst, "train_list.txt") if dirname == "training" else dst
            has_data = os.path.exists(check_file)
            print(f"  {dst} → {target} (valid={valid}, has_data={has_data})")
        else:
            print(f"  ⚠ {dst} is NOT a symlink! (type={'dir' if os.path.isdir(dst) else 'other'})")

    # ── Verify config path ──────────────────────────────────────────────────
    config_path = f"{STYTTS2_DIR}/config_tamil_ft.yml"
    if not os.path.exists(config_path):
        print(f"⚠ Config not found at {config_path}")
    else:
        print(f"✓ Config: {config_path}")

    # ── Verify all pretrained models ────────────────────────────────────────
    for name, path in [
        ("JDC (F0)", f"{STYTTS2_DIR}/Utils/JDC/bst.t7"),
        ("ASR", f"{STYTTS2_DIR}/Utils/ASR/epoch_00080.pth"),
        ("PLBERT", f"{STYTTS2_DIR}/Utils/PLBERT/step_1000000.t7"),
        ("Base weights", f"{TRAINING_DIR}/kokoro_base.pth"),
    ]:
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"⚠ Missing {name}: {path}")

    # ── Run Stage 1 training ────────────────────────────────────────────────
    cmd = ["python3", f"{STYTTS2_DIR}/train_first.py", "-p", config_path]
    if resume_epoch > 0:
        cmd.extend(["--resume_epoch", str(resume_epoch)])

    print(f"\n🚀 Command: {' '.join(cmd)}")
    print(f"   Working dir: {STYTTS2_DIR} (real: {os.path.realpath(STYTTS2_DIR)})")
    print(f"   Config data paths (absolute):")
    print(f"     train_data    → /root/data/training/train_list.txt")
    print(f"     root_path     → /root/data/dataset/audio")
    print(f"     pretrained    → /root/data/training/kokoro_base.pth")
    print(f"     OOD_data      → /root/data/training/OOD_texts.txt")
    print(f"   Relative paths from CWD resolve via kokoro-deutsch/ symlinks:")
    print(f"     ../training/  → {KOKORO_DEUTSCH_DIR}/training → {DATA_DIR}/training")
    print(f"     ../dataset/   → {KOKORO_DEUTSCH_DIR}/dataset → {DATA_DIR}/dataset")
    print(f"     Utils/        → {STYTTS2_DIR}/Utils/")
    print("")

    result = subprocess.run(cmd, cwd=STYTTS2_DIR)

    vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 training failed (exit code {result.returncode})")

    print("✅ Stage 1 training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Stage 2 Training (Duration/SLM model)
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=train_image,
    gpu="A100-80GB",
    volumes={DATA_DIR: vol},
    timeout=86400,     # 24 hours max
    memory=65536,      # 64 GB
    cpu=8,
)
def train_stage2(hf_token: str, resume_epoch: int = 0):
    """Fine-tune Kokoro Stage 2 (duration predictor + SLM adversarial)."""
    import torch
    import os
    import shutil
    import subprocess

    os.environ["HF_TOKEN"] = hf_token

    print(f"🔧 Stage 2 Training")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Resume from epoch: {resume_epoch}")

    # ── Ensure essential symlinks exist ───────────────────────────────────
    for dirname in ["training", "dataset"]:
        src = f"{DATA_DIR}/{dirname}"
        dst = f"{KOKORO_DEUTSCH_DIR}/{dirname}"
        if os.path.islink(dst):
            pass  # symlink exists — OK
        elif os.path.isdir(dst):
            backup = f"{dst}.gitbackup"
            if not os.path.exists(backup):
                os.rename(dst, backup)
                print(f"  Backed up {dst} → {backup}")
            else:
                shutil.rmtree(dst)
                print(f"  Removed existing {dst}")
            os.symlink(src, dst)
            print(f"✓ Created symlink: {dst} → {src}")
        elif not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"✓ Created symlink: {dst} → {src}")

    # Ensure wav symlink exists and points to correct target
    wav_link = f"{STYTTS2_DIR}/wav"
    audio_target = f"{DATASET_DIR}/audio/kavi"
    if os.path.islink(wav_link):
        if os.readlink(wav_link) != audio_target:
            os.unlink(wav_link)
            os.symlink(audio_target, wav_link)
    elif os.path.isdir(wav_link):
        shutil.rmtree(wav_link)
        os.symlink(audio_target, wav_link)
    elif not os.path.exists(wav_link):
        os.symlink(audio_target, wav_link)

    # Build monotonic_align (if needed)
    os.chdir(STYTTS2_DIR)
    try:
        import monotonic_align
        print("✓ monotonic_align already available")
    except ImportError:
        print("⚠ monotonic_align not found, installing via pip...")
        subprocess.run(
            ["pip", "install", "git+https://github.com/resemble-ai/monotonic_align.git"],
            capture_output=True, text=True, check=True,
        )

    # ── Patch config: increase slmadv min_len to prevent Conv2d size error ──
    # After 4 ResBlk halvings (÷16), inputs as small as 5×3 are too small
    # for Conv2d(kernel=5, padding=0). min_len=192 → 192/2/16=6 ≥ 5.
    config_path = f"{STYTTS2_DIR}/config_tamil_ft.yml"
    import re as _re
    with open(config_path, "r") as f:
        config_text = f.read()
    config_text = _re.sub(r'min_len:\s*\d+', 'min_len: 192', config_text)
    with open(config_path, "w") as f:
        f.write(config_text)
    print(f"✓ Patched config: slmadv min_len → 192")

    # ── Patch models.py and slmadv.py using helper functions ──────────────
    models_path = f"{STYTTS2_DIR}/models.py"
    slmadv_path = f"{STYTTS2_DIR}/Modules/slmadv.py"
    
    patch_models_py(models_path)
    patch_slmadv_py(slmadv_path)

    config_path = f"{STYTTS2_DIR}/config_tamil_ft.yml"
    cmd = ["python3", f"{STYTTS2_DIR}/train_second.py", "-p", config_path]
    if resume_epoch > 0:
        cmd.extend(["--resume_epoch", str(resume_epoch)])

    print(f"🚀 Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=STYTTS2_DIR)

    vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Stage 2 training failed (exit code {result.returncode})")

    print("✅ Stage 2 training complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Test inference
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=train_image,
    gpu="A100-80GB",
    volumes={DATA_DIR: vol},
    timeout=1800,     # 30 min
    memory=32768,
    cpu=4,
)
def test_inference(hf_token: str):
    """Generate test samples from fine-tuned model."""
    import os
    import subprocess

    os.environ["HF_TOKEN"] = hf_token

    output_dir = f"{DATA_DIR}/test_output"
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        ["python3", f"{KOKORO_TAMIL_DIR}/scripts/test_inference.py",
         "--output-dir", output_dir],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[:2000])
        raise RuntimeError("Test inference failed")

    vol.commit()
    print("✅ Test inference complete!")


# ─────────────────────────────────────────────────────────────────────────────
# Check progress
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=train_image,
    volumes={DATA_DIR: vol},
    timeout=300,
    memory=4096,
)
def check_progress():
    """Check training progress by reading log files."""
    import os
    import glob

    print("=" * 60)
    print("Checking training progress...")
    print("=" * 60)

    # Check Stage 1 logs
    log_dir = LOGS_DIR
    if os.path.exists(log_dir):
        log_files = sorted(glob.glob(f"{log_dir}/*.log"))
        print(f"\n📁 Log directory: {log_dir}")
        print(f"   Log files: {len(log_files)}")

        for lf in log_files[-3:]:  # Show last 3 log files
            print(f"\n📄 {os.path.basename(lf)}:")
            with open(lf) as f:
                lines = f.readlines()
            # Show last 20 lines
            for line in lines[-20:]:
                print(f"   {line.rstrip()}")
    else:
        print(f"⚠ No log directory found at {log_dir}")

    # Check checkpoints
    ckpt_pattern = f"{log_dir}/*.pth"
    checkpoints = glob.glob(ckpt_pattern) if os.path.exists(log_dir) else []
    print(f"\n💾 Checkpoints found: {len(checkpoints)}")
    for ckpt in checkpoints:
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        print(f"   {os.path.basename(ckpt)}: {size_mb:.1f} MB")

    # Check dataset
    train_list = f"{TRAINING_DIR}/train_list.txt"
    val_list = f"{TRAINING_DIR}/val_list.txt"
    for name in [train_list, val_list]:
        if os.path.exists(name):
            with open(name) as f:
                count = sum(1 for _ in f)
            print(f"   {os.path.basename(name)}: {count} entries")

    vol.commit()
    return {
        "log_files": len(glob.glob(f"{log_dir}/*.log")) if os.path.exists(log_dir) else 0,
        "checkpoints": len(checkpoints),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    stage: str = "1",
    hf_token: str = "",
    skip_download: bool = False,
    skip_prepare: bool = False,
    skip_setup: bool = False,
    resume_epoch: int = 0,
    test_only: bool = False,
):
    """Launch the Kokoro Tamil fine-tuning pipeline on Modal."""
    import os

    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("❌ HF_TOKEN required. Set env var or pass --hf-token.")
        print("   This is needed for downloading private datasets and WavLM model.")
        sys.exit(1)

    print("=" * 60)
    print("🎵 Kokoro-82M Tamil Fine-Tuning — Modal Pipeline")
    print("=" * 60)
    print(f"  Stage: {stage}")
    print(f"  Skip download: {skip_download}")
    print(f"  Skip prepare: {skip_prepare}")
    print(f"  Skip setup: {skip_setup}")
    print(f"  Resume epoch: {resume_epoch}")

    if test_only:
        print("\n🧪 Running test inference only...")
        test_inference.remote(hf_token=hf_token)
        # Also check progress
        prog = check_progress.remote()
        print(f"\nProgress: {prog}")
        return

    # Step 1: Download data
    if not skip_download:
        print("\n⬇ [1/5] Downloading datasets & base model...")
        download_data.remote(hf_token=hf_token)
    else:
        print("\n⏭ [1/5] Skipping download (using existing volume data)")

    # Step 2: Prepare data
    if not skip_prepare:
        print("\n🔄 [2/5] Preparing training data...")
        prepare_data.remote()
    else:
        print("\n⏭ [2/5] Skipping data preparation")

    # Step 3: Setup training environment
    if not skip_setup:
        print("\n🔧 [3/5] Setting up training environment...")
        setup_training.remote(hf_token=hf_token)
    else:
        print("\n⏭ [3/5] Skipping setup")

    # Step 4: Stage 1 training
    if stage in ("1", "both"):
        print("\n🚀 [4/5] Starting Stage 1 training (acoustic/prosody)...")
        train_stage1.remote(hf_token=hf_token, resume_epoch=resume_epoch)

    # Step 5: Stage 2 training
    if stage in ("2", "both"):
        # Verify Stage 1 checkpoint exists before starting Stage 2
        # Note: can't check volume contents from local entrypoint, so just warn
        print("\n🚀 [5/5] Starting Stage 2 training (duration/SLM)...")
        print("⚠ Make sure first_stage.pth exists on the volume from Stage 1!")
        train_stage2.remote(hf_token=hf_token, resume_epoch=resume_epoch)

    print("\n🎉 Done!")