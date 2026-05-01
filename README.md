# Kokoro-82M Tamil Fine-Tuning

Fine-tuning [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) for Tamil TTS using manually transcribed audio data.

## Architecture

Based on the [kokoro-deutsch](https://github.com/semidark/kokoro-deutsch) training recipe:
- **Kokoro-82M**: 82M param TTS model (StyleTTS2 + iSTFTNet)
- **Training**: Patched StyleTTS2 with 2-stage training (acoustic → prosody)
- **Data**: ~30K Tamil audio-transcription pairs (8K manually verified)
- **G2P**: Custom Tamil → IPA phoneme mapping (espeak-ng Tamil is poor quality)

## Quick Start

### 1. Set up environment (on GPU machine)

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/semidark/kokoro-deutsch
cd kokoro-deutsch
uv sync

# Install StyleTTS2 dependencies
cd StyleTTS2
pip install torch torchaudio accelerate transformers librosa soundfile pyyaml tensorboard munch phonemizer huggingface_hub Cython
python setup.py build_ext --inplace  # monotonic_align
cd ..

# Install espeak-ng (for G2P)
sudo apt-get install espeak-ng libsndfile1
```

### 2. Download datasets from HuggingFace

```bash
export HF_TOKEN=your_token_here
python scripts/download_datasets.py --token $HF_TOKEN --output ./dataset
```

### 3. Prepare training data

```bash
python scripts/prepare_tamil_data.py prepare
python scripts/prepare_tamil_data.py verify
```

### 4. Convert Kokoro weights

```bash
python scripts/prepare_tamil_data.py convert-weights
python scripts/prepare_tamil_data.py patch-styletts2
```

### 5. Train

```bash
# Stage 1: Acoustic model
cd StyleTTS2
accelerate launch train_first.py --config_path ../configs/config_tamil_ft.yml

# Stage 2: Prosody/duration
accelerate launch train_second.py --config_path ../configs/config_tamil_ft.yml
```

### 6. Extract voicepack & test

```bash
python scripts/extract_voicepack.py \
  --model StyleTTS2/logs/kokoro-tamil/epoch_2nd_00010.pth \
  --audio-dir dataset/audio/kavi \
  --output voices/tamil_kavi.pt

python scripts/test_inference.py \
  --checkpoint StyleTTS2/logs/kokoro-tamil/epoch_2nd_00010.pth \
  --voicepack voices/tamil_kavi.pt \
  --output-dir test_output/
```

## Dataset Summary

| Dataset | Examples | Size | Format |
|---|---|---|---|
| kavi-ps-manual-10hr-original | 8,023 | 1.55 GB | audio + transcription (verified) |
| ps-transcribed (ps-1..ps-4) | ~21K | 18.4 GB | audio + transcription + timestamps |
| tts-1000-v1 | 1,000 | 193 MB | audio + transcription (public) |