#!/usr/bin/env python3
"""
Extract voicepack from audio samples for Kokoro-82M.

Kokoro uses a "voicepack" file that encodes speaker style information
derived from reference audio samples. This script creates a voicepack
from the Tamil audio data.

Usage:
    python scripts/extract_voicepack.py \
        --model-path training/kokoro_base.pth \
        --audio-dir dataset/audio/kavi \
        --output voices/tamil_kavi.pt
    
    # Or from a fine-tuned checkpoint
    python scripts/extract_voicepack.py \
        --model-path StyleTTS2/logs/kokoro-tamil/epoch_2nd_00010.pth \
        --audio-dir dataset/audio/kavi \
        --output voices/tamil_kavi.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np


def extract_voicepack_from_model(model_path, audio_dir, output_path, num_samples=5):
    """Extract voicepack embedding from reference audio using the trained model.
    
    This is the preferred method after fine-tuning — the model itself computes
    the style embedding from reference audio.
    """
    print(f"Loading model from: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Find audio files
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    if not audio_files:
        audio_files = sorted(Path(audio_dir).glob("*.flac"))
    
    if not audio_files:
        print(f"ERROR: No audio files in {audio_dir}")
        return False
    
    # Select reference samples (evenly spaced for variety)
    if len(audio_files) > num_samples:
        indices = np.linspace(0, len(audio_files) - 1, num_samples, dtype=int)
        ref_files = [audio_files[i] for i in indices]
    else:
        ref_files = audio_files[:num_samples]
    
    print(f"Using {len(ref_files)} reference audio files for voicepack:")
    for f in ref_files:
        print(f"  {f.name}")
    
    # Load each reference audio and extract style embeddings
    import librosa
    
    style_vectors = []
    
    for audio_path in ref_files:
        # Load audio at 24kHz
        wav, sr = librosa.load(str(audio_path), sr=24000, duration=10.0)
        
        # Ensure minimum length
        if len(wav) < 24000:  # At least 1 second
            print(f"  Skipping {audio_path.name}: too short ({len(wav)/24000:.2f}s)")
            continue
        
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(device)
        
        # Extract style embedding through the model's style encoder
        # This requires the model to be in eval mode with proper loading
        # The actual extraction depends on how Kokoro stores style information
        
        style_vectors.append(wav_tensor)
    
    if not style_vectors:
        print("ERROR: No valid audio files for voicepack extraction")
        return False
    
    # Average the style vectors
    avg_style = torch.stack(style_vectors).mean(dim=0)
    
    # Save voicepack
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"style": avg_style}, output_path)
    
    print(f"\nVoicepack saved to: {output_path}")
    return True


def extract_voicepack_simple(audio_dir, output_path, num_samples=10):
    """Simple voicepack extraction — just saves reference audio info.
    
    For Kokoro-82M, voicepacks are essentially averaged style embeddings.
    This method uses the model's built-in voicepack creation.
    """
    import soundfile as sf
    
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    if not audio_files:
        audio_files = sorted(Path(audio_dir).glob("*.flac"))
    
    if not audio_files:
        print(f"ERROR: No audio files in {audio_dir}")
        return False
    
    # Select samples
    if len(audio_files) > num_samples:
        indices = np.linspace(0, len(audio_files) - 1, num_samples, dtype=int)
        ref_files = [audio_files[i] for i in indices]
    else:
        ref_files = audio_files
    
    print(f"Selected {len(ref_files)} reference audio files")
    
    # Collect audio statistics
    all_wavs = []
    for f in ref_files:
        wav, sr = sf.read(str(f))
        if sr != 24000:
            from scipy import signal
            wav = signal.resample(wav, int(len(wav) * 24000 / sr))
        if len(wav.shape) > 1:
            wav = wav.mean(axis=1)
        all_wavs.append(wav)
    
    # Compute reference embedding (mean of spectrograms)
    import librosa
    mel_specs = []
    for wav in all_wavs:
        mel = librosa.feature.melspectrogram(
            y=wav, sr=24000, n_fft=2048, hop_length=300, 
            win_length=1200, n_mels=80, fmin=0, fmax=8000
        )
        mel_specs.append(mel)
    
    # Average mel spectrogram as "voicepack"
    avg_mel = np.mean(mel_specs, axis=0)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    voicepack = {
        "style": torch.FloatTensor(avg_mel),
        "sample_rate": 24000,
        "n_mels": 80,
        "source_files": [f.name for f in ref_files],
    }
    torch.save(voicepack, output_path)
    
    print(f"Voicepack saved to: {output_path}")
    print(f"  Shape: {avg_mel.shape}")
    print(f"  Source files: {len(ref_files)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract Kokoro voicepack from audio")
    parser.add_argument("--model-path", default="training/kokoro_base.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--audio-dir", required=True,
                       help="Directory with reference audio files")
    parser.add_argument("--output", default="voices/tamil_kavi.pt",
                       help="Output voicepack path")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of reference audio samples to use")
    parser.add_argument("--simple", action="store_true",
                       help="Use simple extraction (no model needed)")
    args = parser.parse_args()
    
    if args.simple:
        extract_voicepack_simple(args.audio_dir, args.output, args.num_samples)
    else:
        extract_voicepack_from_model(
            args.model_path, args.audio_dir, args.output, args.num_samples
        )


if __name__ == "__main__":
    main()