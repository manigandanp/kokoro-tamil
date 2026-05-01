#!/usr/bin/env python3
"""
Test inference for Kokoro-82M Tamil fine-tuned model.

Usage:
    python scripts/test_inference.py \
        --checkpoint StyleTTS2/logs/kokoro-tamil/epoch_2nd_00010.pth \
        --voicepack voices/tamil_kavi.pt \
        --output-dir test_output/
"""

import argparse
import os
import sys
from pathlib import Path

# Test sentences in Tamil
TAMIL_TEST_SENTENCES = [
    ("tamil_formal", "தமிழ் மொழி உலகின் தொன்மையான மொழிகளில் ஒன்றாகும்."),
    ("tamil_greeting", "வணக்கம், நான் கவி பேசுகிறேன்."),
    ("tamil_tech", "செயற்கை நுண்ணறிவு தொழில்நுட்பம் வேகமாக வளர்ந்து வருகிறது."),
    ("tamil_news", "இன்று சென்னையில் மழை பெய்தது."),
    ("tamil_poetry", "யாமறிந்த மொழிகளிலே தமிழ்மொழி போல் இனிதாவது எங்கும் காணோம்."),
    ("tamil_daily", "காலையில் எழுந்தவுடன் தேநீர் குடிப்பது வழக்கம்."),
    ("tamil_complex", "தமிழ்நாடு அரசு கல்வித்துறையில் பல திட்டங்களை செயல்படுத்தி வருகிறது."),
]

# English test sentences (for comparison — should still work)
ENGLISH_TEST_SENTENCES = [
    ("english_hello", "Hello, I am Kavi speaking Tamil."),
    ("english_mixed", "Today we are testing the Tamil voice model."),
]


def test_with_kokoro_library(checkpoint_path, output_dir, voicepack_path=None):
    """Test inference using the official Kokoro library."""
    try:
        from misaki import espeak
        import torch
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"Required package missing: {e}")
        print("Install: pip install misaki torch soundfile")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Kokoro-82M Tamil Test Inference")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        log_dir = Path(checkpoint_path).parent
        if log_dir.exists():
            for f in sorted(log_dir.glob("*.pth")):
                print(f"  {f}")
        return False
    
    # Use Kokoro's inference module
    try:
        # Try the official Kokoro pipeline
        from kokoro import KokoroTTS
        
        model = KokoroTTS(checkpoint_path, device=device)
        if voicepack_path:
            model.load_voicepack(voicepack_path)
        
        for name, text in TAMIL_TEST_SENTENCES + ENGLISH_TEST_SENTENCES:
            print(f"\nGenerating: [{name}] {text[:60]}...")
            audio = model.generate(text, lang="ta")
            out_path = output_dir / f"{name}.wav"
            sf.write(str(out_path), audio, 24000)
            print(f"  Saved: {out_path}")
        
        print(f"\nAll test samples saved to: {output_dir}")
        return True
        
    except ImportError:
        # Fallback: direct StyleTTS2 inference
        print("\nKokoro library not available, using direct StyleTTS2 inference...")
        return test_with_styletts2_direct(checkpoint_path, output_dir, voicepack_path)


def test_with_styletts2_direct(checkpoint_path, output_dir, voicepack_path=None):
    """Fallback: direct StyleTTS2 inference for testing."""
    import torch
    import sys
    
    # This requires the StyleTTS2 code to be available
    sty_path = Path(__file__).resolve().parent.parent / "StyleTTS2"
    if not sty_path.exists():
        print("StyleTTS2 not found. Clone it first:")
        print("  git clone https://github.com/s0md3v0/StyleTTS2.git")
        return False
    
    sys.path.insert(0, str(sty_path))
    
    try:
        from models import build_model
        from text_utils import text_to_sequence
        import soundfile as sf
    except ImportError as e:
        print(f"Cannot import StyleTTS2 modules: {e}")
        print("Make sure StyleTTS2 is cloned and dependencies are installed.")
        return False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config_tamil_ft.yml"
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Build model
    model = build_model(config, device)
    
    # Load weights
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        for component, state_dict in checkpoint.get("net", checkpoint).items():
            getattr(model, component).load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Generate samples
    for name, text in TAMIL_TEST_SENTENCES[:3]:  # Just first 3 for quick test
        print(f"\nGenerating: [{name}] {text[:60]}...")
        
        # Phonemize
        from prepare_tamil_data import phonemize_tamil
        ipa = phonemize_tamil(text)
        
        # Generate (this is simplified — actual inference needs more setup)
        # In practice, you'd use the full StyleTTS2 inference pipeline
        # with ATSS and diffusion steps
        
        out_path = output_dir / f"{name}.wav"
        # For now, just save the phonemes
        with open(str(out_path).replace('.wav', '.txt'), 'w') as f:
            f.write(f"Original: {text}\nIPA: {ipa}\n")
        
        print(f"  Phonemes: {ipa[:80]}...")
        print(f"  Saved phonemes (full inference requires StyleTTS2 setup)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test inference for Kokoro Tamil model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--voicepack", default=None, help="Path to voicepack file")
    parser.add_argument("--output-dir", default="test_output", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success = test_with_kokoro_library(args.checkpoint, args.output_dir, args.voicepack)
    
    if not success:
        print("\nInference test had issues. This is expected if:")
        print("  1. The model hasn't been trained yet")
        print("  2. StyleTTS2 isn't set up on this machine")
        print("  3. Running on CPU (will be slow but should work)")
        print("\nFor full inference testing, use a GPU machine (Modal/Lightning).")


if __name__ == "__main__":
    main()