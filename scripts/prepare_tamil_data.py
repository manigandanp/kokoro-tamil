#!/usr/bin/env python3
"""
Kokoro Tamil: Prepare Training Data
=====================================

Converts Tamil TTS datasets from HuggingFace into the format 
expected by StyleTTS2's training scripts (kokoro-deutsch pipeline).

Adapted from kokoro-deutsch/scripts/prepare_training.py for Tamil.

Usage:
    python scripts/prepare_tamil_data.py prepare
    python scripts/prepare_tamil_data.py convert-weights
    python scripts/prepare_tamil_data.py patch-styletts2
    python scripts/prepare_tamil_data.py verify
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
AUDIO_DIR = DATASET_DIR / "audio"
TRAINING_DIR = ROOT_DIR / "training"

TRAIN_LIST = TRAINING_DIR / "train_list.txt"
VAL_LIST = TRAINING_DIR / "val_list.txt"
OOD_FILE = TRAINING_DIR / "OOD_texts.txt"

# ── Audio params (must match Kokoro/StyleTTS2 config) ────────────────────────

SAMPLE_RATE = 24000
N_FFT = 2048
HOP_LENGTH = 300
WIN_LENGTH = 1200
N_MELS = 80
F_MIN = 0
F_MAX = 8000

# ── Split ────────────────────────────────────────────────────────────────────

VAL_RATIO = 0.05
RANDOM_SEED = 42

# ── Filtering ────────────────────────────────────────────────────────────────

MIN_DURATION_S = 1.5
MAX_DURATION_S = 30.0
MIN_PHONEMES = 5


# ─────────────────────────────────────────────────────────────────────────────
# Tamil G2P: Custom Tamil → IPA mapping
# ─────────────────────────────────────────────────────────────────────────────

# Tamil Unicode range: U+0B80 - U+0BFF
# This maps Tamil graphemes to IPA phonemes compatible with Kokoro's 178-token vocabulary

TAMIL_VOWELS = {
    # Short vowels
    'அ': 'ʌ',       # a
    'ஆ': 'aː',      # aa (long)
    'இ': 'i',       # i
    'ஈ': 'iː',      # ii (long)
    'உ': 'u',       # u
    'ஊ': 'uː',      # uu (long)
    'எ': 'e',       # e
    'ஏ': 'eː',      # ee (long)
    'ஐ': 'aɪ',     # ai
    'ஒ': 'o',       # o 
    'ஓ': 'oː',      # oo (long)
    'ஔ': 'aʊ',     # au
    
    # Vowel diacritics (consonant + vowel combinations handled separately)
}

TAMIL_CONSONANTS = {
    'க': 'k',       # ka - can be k, g, x depending on position
    'ங': 'ŋ',       # nga
    'ச': 'ʧ',      # cha - contextual, often s initially
    'ஜ': 'dʒ',     # ja (Grantha)
    'ஞ': 'ɲ',       # nya
    'ட': 'ʈ',       # ṭa (retroflex)
    'ண': 'ɳ',       # ṇa (retroflex nasal)
    'த': 't̪',      # tha (dental)
    'ந': 'n̪',      # na (dental)
    'ன': 'n',       # na (alveolar)
    'ப': 'p',       # pa
    'ம': 'm',       # ma
    'ய': 'j',       # ya
    'ர': 'r',       # ra
    'ற': 'ɻ',       # ṟa (retroflex approximant)
    'ல': 'l',       # la
    'ள': 'ɭ',       # ḷa (retroflex lateral)
    'ழ': 'ɻ',       # zha (retroflex approximant, like ற)
    'வ': 'ʋ',       # va
    'ஶ': 'ʃ',       # sha (Grantha)
    'ஷ': 'ʃ',       # sha (Grantha)  
    'ஸ': 's',       # sa (Grantha)
    'ஹ': 'h',       # ha (Grantha)
    'க்ஷ': 'kʃ',   # ksha
}

# Vowel signs (diacritics) - appended to consonants
TAMIL_VOWEL_SIGNS = {
    'ா': 'aː',      # aa mark
    'ி': 'i',        # i mark
    'ீ': 'iː',      # ii mark
    'ு': 'u',        # u mark
    'ூ': 'uː',      # uu mark
    'ெ': 'e',        # e mark
    'ே': 'eː',   # ee mark
    'ை': 'aɪ',   # ai mark
    'ொ': 'o',     # o mark
    'ோ': 'oː',   # oo mark
    'ௌ': 'aʊ',   # au mark
}

# Pulli (virama) - consonant without vowel
TAMIL_PULLI = '்'  # indicates consonant with no following vowel


def tamil_to_ipa(text: str) -> str:
    """Convert Tamil text to IPA phonemes.
    
    This is a simplified rule-based converter. For production use,
    consider using espeak-ng as a fallback or validator.
    """
    ipa = []
    i = 0
    text = text.strip()
    
    while i < len(text):
        char = text[i]
        
        # Skip punctuation, digits, spaces for now
        if char in ' ,.;:!?।॥\n\r\t' or char.isdigit():
            if char in '!?।':
                ipa.append(' ')
            i += 1
            continue
        
        # Check for multi-char consonant clusters first
        if i + 1 < len(text):
            two_char = text[i:i+2]
            if two_char in TAMIL_CONSONANTS:
                ipa.append(TAMIL_CONSONANTS[two_char])
                i += 2
                continue
        
        # Standalone vowels (at word start or after space)
        if char in TAMIL_VOWELS:
            ipa.append(TAMIL_VOWELS[char])
            i += 1
            continue
        
        # Consonants
        if char in TAMIL_CONSONANTS:
            consonant = TAMIL_CONSONANTS[char]
            # Look ahead for vowel sign or pulli
            if i + 1 < len(text):
                next_char = text[i + 1]
                if next_char == TAMIL_PULLI:
                    # Consonant without vowel (consonant cluster)
                    ipa.append(consonant)
                    i += 2
                    continue
                elif next_char in TAMIL_VOWEL_SIGNS:
                    # Consonant + vowel sign
                    ipa.append(consonant + TAMIL_VOWEL_SIGNS[next_char])
                    i += 2
                    continue
            # Default: consonant with inherent vowel 'a' (ʌ)
            ipa.append(consonant + 'ʌ')
            i += 1
            continue
        
        # Vowel signs (shouldn't reach here normally, as they're handled with consonants)
        if char in TAMIL_VOWEL_SIGNS:
            ipa.append(TAMIL_VOWEL_SIGNS[char])
            i += 1
            continue
        
        # Pulli (handled with consonant above, but just in case)
        if char == TAMIL_PULLI:
            i += 1
            continue
        
        # Unknown character - try espeak-ng fallback
        ipa.append(char)
        i += 1
    
    result = ''.join(ipa)
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def phonemize_tamil_espeak(text: str) -> str:
    """Fallback phonemization using espeak-ng (lower quality for Tamil)."""
    try:
        from misaki import espeak
        g2p = espeak.EspeakG2P(language='ta')
        phonemes, _ = g2p(text)
        return phonemes
    except ImportError:
        return tamil_to_ipa(text)


def phonemize_tamil(text: str, use_espeak: bool = True) -> str:
    """Convert Tamil text to IPA phonemes.
    
    Uses custom Tamil G2P first, with espeak-ng as fallback.
    """
    # First try custom G2P
    custom_ipa = tamil_to_ipa(text)
    
    if use_espeak:
        try:
            espeak_ipa = phonemize_tamil_espeak(text)
            # Use espeak if custom produces very short output
            if len(espeak_ipa) > len(custom_ipa) * 1.5:
                return espeak_ipa
        except Exception:
            pass
    
    return custom_ipa


# ─────────────────────────────────────────────────────────────────────────────
# Tamil OOD (Out-of-Domain) texts for training
# ─────────────────────────────────────────────────────────────────────────────

TAMIL_OOD_SENTENCES = [
    "தமிழ் மொழி உலகின் தொன்மையான மொழிகளில் ஒன்றாகும்.",
    "இந்தியா ஒரு பன்மொழி நாடு ஆகும்.",
    "சென்னை தமிழ்நாட்டின் தலைநகரம் ஆகும்.",
    "அறிவியல் மற்றும் தொழில்நுட்பம் நாட்டின் முன்னேற்றத்திற்கு அவசியம்.",
    "கல்வி என்பது வாழ்வின் அடிப்படை ஆகும்.",
    "இயற்கை அழகு மனதை மகிழ்விக்கிறது.",
    "நாம் ஒற்றுமையாக செயல்பட வேண்டும்.",
    "விவசாயம் இந்தியாவின் முக்கிய தொழில் ஆகும்.",
    "மழைக்காலத்தில் ஆறுகள் நிரம்பி வழிகின்றன.",
    "புத்தகம் வாசிப்பது நல்ல பழக்கம்.",
    "ஒவ்வொரு மனிதனுக்கும் சமமான உரிமை உண்டு.",
    "தொழில்நுட்பம் வாழ்க்கையை எளிதாக்குகிறது.",
    "மருத்துவம் மற்றும் சுகாதாரம் முக்கியமானவை.",
    "கிராமப்புறங்களில் விவசாயம் முக்கிய தொழில்.",
    "கலை மற்றும் கலாச்சாரம் நாகரிகத்தின் அடையாளங்கள்.",
    "காலையில் எழுந்தவுடன் உடற்பயிற்சி செய்வது நல்லது.",
    "இணையதளம் தகவல் பரிமாற்றத்தை எளிதாக்கியுள்ளது.",
    "சென்னை கடற்கரை மிகவும் அழகாக இருக்கும்.",
    "தமிழ் இலக்கியம் உலகப் புகழ்பெற்றது.",
    "கல்வியின் வழியே நாடு முன்னேறும்.",
]


def load_hf_dataset_parquet(path: Path, split="train"):
    """Load a parquet-based HF dataset from local files."""
    import pyarrow.parquet as pq
    
    data = []
    parquet_files = sorted(path.glob("**/*.parquet"))
    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()
        data.extend(df.to_dict('records'))
    return data


def process_kavi_dataset(raw_dir: Path, audio_dir: Path) -> list:
    """Process kavi-ps-manual-10hr-original dataset."""
    dataset_path = raw_dir / "kavi-ps-manual-10hr-original"
    
    entries = []
    
    # Try loading from parquet files
    parquet_dir = dataset_path / "data"
    if parquet_dir.exists():
        import pyarrow.parquet as pq
        for pf in sorted(parquet_dir.glob("*.parquet")):
            table = pq.read_table(pf)
            df = table.to_pandas()
            for _, row in df.iterrows():
                audio = row.get('audio', {})
                transcription = row.get('transcription', '')
                file_name = row.get('file_name', '')
                
                if not transcription:
                    continue
                
                ipa = phonemize_tamil(transcription)
                if len(ipa) < MIN_PHONEMES:
                    continue
                
                # We'll extract audio later
                entries.append({
                    'source': 'kavi',
                    'file_name': file_name,
                    'text': transcription,
                    'ipa': ipa,
                    'speaker': 'kavi',
                    'audio_data': audio,
                })
    
    return entries


def process_ps_transcribed(raw_dir: Path, audio_dir: Path) -> list:
    """Process ps-transcribed dataset (multiple configs)."""
    entries = []
    
    dataset_path = raw_dir / "ps-transcribed"
    
    for config in ["ps-1", "ps-2", "ps-3", "ps-4"]:
        config_dir = dataset_path / config
        if not config_dir.exists():
            # Try alternative path structure
            config_dir = dataset_path
        
        parquet_files = sorted(list(dataset_path.glob(f"{config}/**/*.parquet")) + 
                              list(dataset_path.glob(f"{config}/*.parquet")))
        
        for pf in parquet_files:
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(pf)
                df = table.to_pandas()
                for _, row in df.iterrows():
                    audio = row.get('audio', {})
                    transcription = row.get('transcription', '')
                    file_name = row.get('chunk_name', row.get('file_name', ''))
                    duration = row.get('duration', 0)
                    
                    if not transcription or not isinstance(transcription, str):
                        continue
                    
                    if duration and (duration < MIN_DURATION_S or duration > MAX_DURATION_S):
                        continue
                    
                    ipa = phonemize_tamil(transcription)
                    if len(ipa) < MIN_PHONEMES:
                        continue
                    
                    speaker = row.get('speaker', 'kavi')
                    if not isinstance(speaker, str):
                        speaker = 'kavi'
                    
                    entries.append({
                        'source': f'ps-{config}',
                        'file_name': file_name,
                        'text': transcription,
                        'ipa': ipa,
                        'speaker': speaker,
                        'duration': duration,
                        'audio_data': audio,
                    })
            except Exception as e:
                print(f"  Warning: Error processing {pf}: {e}")
                continue
    
    return entries


def process_tts1000(raw_dir: Path, audio_dir: Path) -> list:
    """Process tts-1000-v1 public dataset."""
    dataset_path = raw_dir / "tts-1000-v1"
    entries = []
    
    parquet_dir = dataset_path / "data"
    if parquet_dir.exists():
        import pyarrow.parquet as pq
        for pf in sorted(parquet_dir.glob("*.parquet")):
            table = pq.read_table(pf)
            df = table.to_pandas()
            for _, row in df.iterrows():
                transcription = row.get('transcription', '')
                file_name = row.get('file_name', '')
                audio = row.get('audio', {})
                
                if not transcription:
                    continue
                
                ipa = phonemize_tamil(transcription)
                if len(ipa) < MIN_PHONEMES:
                    continue
                
                entries.append({
                    'source': 'tts-1000',
                    'file_name': file_name,
                    'text': transcription,
                    'ipa': ipa,
                    'speaker': 'kavi',
                    'audio_data': audio,
                })
    
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────


def cmd_prepare():
    """Generate train/val lists and metadata from downloaded datasets."""
    print("=" * 60)
    print("Kokoro Tamil: Preparing Training Data")
    print("=" * 60)
    
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    all_entries = []
    
    # Process each dataset
    datasets = [
        ("kavi-ps-manual-10hr-original", process_kavi_dataset),
        ("ps-transcribed", process_ps_transcribed),  
        ("tts-1000-v1", process_tts1000),
    ]
    
    for name, processor in datasets:
        print(f"\nProcessing {name}...")
        try:
            entries = processor(RAW_DIR, AUDIO_DIR)
            print(f"  Found {len(entries)} valid entries")
            all_entries.extend(entries)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not all_entries:
        print("\nERROR: No entries found. Did you run download_datasets.py first?")
        print("Looking in:", RAW_DIR)
        sys.exit(1)
    
    print(f"\nTotal entries: {len(all_entries)}")
    
    # Deduplicate by text content
    seen = set()
    unique_entries = []
    for entry in all_entries:
        if entry['text'] not in seen:
            seen.add(entry['text'])
            unique_entries.append(entry)
    
    print(f"After deduplication: {len(unique_entries)}")
    
    # Train/val split
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(unique_entries)
    n_val = max(1, int(len(unique_entries) * VAL_RATIO))
    val_entries = unique_entries[:n_val]
    train_entries = unique_entries[n_val:]
    
    print(f"Split: {len(train_entries):,} train / {len(val_entries):,} val")
    
    # Write training lists
    # Format: wav_path|ipa_phonemes|speaker_id
    
    # NOTE: Since audio is embedded in parquet, we need to extract WAVs first
    # For now, write metadata and the extraction will happen in a separate step
    metadata_path = TRAINING_DIR / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for entry in train_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\nWrote metadata to {metadata_path}")
    print(f"  Train: {len(train_entries):,} entries")
    print(f"  Val: {len(val_entries):,} entries")
    
    # Write OOD texts
    ood_phonemes = []
    for text in TAMIL_OOD_SENTENCES:
        ipa = phonemize_tamil(text)
        ood_phonemes.append(ipa)
    
    with open(OOD_FILE, 'w') as f:
        f.write('\n'.join(ood_phonemes) + '\n')
    
    print(f"Wrote {OOD_FILE} ({len(ood_phonemes)} sentences)")
    
    # Print statistics
    speakers = set(e['speaker'] for e in unique_entries)
    sources = set(e['source'] for e in unique_entries)
    print(f"\nSpeakers: {speakers}")
    print(f"Sources: {sources}")
    
    # Sample phonemes
    print(f"\nSample phonemes (first 5):")
    for i, entry in enumerate(unique_entries[:5]):
        print(f"  [{entry['source']}] {entry['text'][:60]}...")
        print(f"    IPA: {entry['ipa'][:80]}...")
    
    print(f"\n{'='*60}")
    print(f"NEXT STEP: Extract WAV audio from parquet files")
    print(f"  python scripts/prepare_tamil_data.py extract-audio")
    print(f"{'='*60}")


def cmd_extract_audio():
    """Extract WAV audio from parquet files and convert to 24kHz mono."""
    import soundfile as sf
    import numpy as np
    
    metadata_path = TRAINING_DIR / "metadata.jsonl"
    if not metadata_path.exists():
        print("ERROR: metadata.jsonl not found. Run 'prepare' first.")
        sys.exit(1)
    
    entries = []
    with open(metadata_path) as f:
        for line in f:
            entries.append(json.loads(line.strip()))
    
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    extracted = 0
    errors = 0
    
    for entry in entries:
        audio_data = entry.get('audio_data', {})
        if not audio_data or 'bytes' not in audio_data:
            # Try to extract from the raw parquet data
            continue
        
        output_name = entry['file_name'] if entry['file_name'] else f"sample_{extracted}.wav"
        output_path = AUDIO_DIR / output_name
        
        if output_path.exists():
            extracted += 1
            continue
        
        try:
            # Decode audio from bytes
            import io
            audio_bytes = audio_data['bytes']
            waveform, sr = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            
            # Resample to 24kHz if needed
            if sr != SAMPLE_RATE:
                from scipy import signal as sig
                num_samples = int(len(waveform) * SAMPLE_RATE / sr)
                waveform = sig.resample(waveform, num_samples)
            
            # Save as 24kHz mono 16-bit WAV
            sf.write(str(output_path), waveform, SAMPLE_RATE, subtype='PCM_16')
            extracted += 1
        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"  Error extracting {output_name}: {e}")
    
    print(f"Extracted: {extracted}, Errors: {errors}")


def cmd_convert_weights(force: bool = False):
    """Convert Kokoro-82M weights to StyleTTS2 format."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required. Install: pip install torch")
        sys.exit(1)

    from huggingface_hub import hf_hub_download

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TRAINING_DIR / "kokoro_base.pth"

    if output_path.exists() and not force:
        print(f"Converted weights already exist: {output_path}")
        print("Use --force to regenerate.")
        return

    print("Downloading Kokoro-82M weights from HuggingFace...")
    model_path = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v1_0.pth")
    config_path = hf_hub_download("hexgrad/Kokoro-82M", "config.json")

    print(f"Loading weights from {model_path}...")
    kokoro_state = torch.load(model_path, map_location="cpu", weights_only=True)

    net = {}
    total_params = 0
    for component, state_dict in kokoro_state.items():
        cleaned = {}
        for key, tensor in state_dict.items():
            clean_key = key.removeprefix("module.")
            cleaned[clean_key] = tensor
            total_params += tensor.numel()
        net[component] = cleaned
        print(f"  {component}: {len(cleaned)} tensors")

    checkpoint = {"net": net}
    import shutil
    config_out = TRAINING_DIR / "config.json"
    shutil.copy2(config_path, config_out)

    torch.save(checkpoint, output_path)
    print(f"\nSaved converted weights: {output_path}")
    print(f"Saved config: {config_out}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")


def cmd_patch_styletts2():
    """Generate Kokoro-compatible symbol mapping for StyleTTS2."""
    config_path = TRAINING_DIR / "config.json"
    if not config_path.exists():
        print("ERROR: training/config.json not found. Run 'convert-weights' first.")
        sys.exit(1)

    from scripts.prepare_training import _generate_symbols_code
    # Reuse the function from kokoro-deutsch's prepare_training.py
    # This reads config.json and generates kokoro_symbols.py
    
    # For now, copy from the reference repo
    print("Patching StyleTTS2 with Kokoro's 178-token vocabulary...")
    print("This requires running prepare_training.py patch-styletts2 from kokoro-deutsch")
    print("\nManual steps:")
    print("  1. cp training/config.json training/kokoro_symbols.py -> StyleTTS2/")
    print("  2. Replace symbols in StyleTTS2/text_utils.py with Kokoro's vocab")
    print("  3. Verify: len(symbols) == 178")


def cmd_verify():
    """Verify training data integrity."""
    print("Verifying training data...")
    
    issues = []
    
    for f in [TRAIN_LIST, VAL_LIST]:
        if not f.exists():
            issues.append(f"MISSING: {f}")
    
    if TRAIN_LIST.exists():
        with open(TRAIN_LIST) as f:
            lines = f.readlines()
        print(f"  Train entries: {len(lines):,}")
        
        # Check format
        bad = 0
        for line in lines[:10]:
            parts = line.strip().split('|')
            if len(parts) != 3:
                bad += 1
        if bad:
            issues.append(f"Bad format in train_list.txt ({bad}/10 checked)")
    
    if VAL_LIST.exists():
        with open(VAL_LIST) as f:
            lines = f.readlines()
        print(f"  Val entries: {len(lines):,}")
    
    # Check weights
    weights_path = TRAINING_DIR / "kokoro_base.pth"
    if weights_path.exists():
        import torch
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        print(f"  Base weights: OK ({list(state.keys())})")
    else:
        print(f"  Base weights: NOT FOUND (run 'convert-weights')")
    
    if issues:
        print("\nISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n  All checks passed!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Tamil training data for Kokoro fine-tuning",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Generate train/val lists from dataset")
    subparsers.add_parser("extract-audio", help="Extract WAV audio from parquet files")
    p_convert = subparsers.add_parser("convert-weights", help="Convert Kokoro weights")
    p_convert.add_argument("--force", action="store_true")
    subparsers.add_parser("patch-styletts2", help="Patch StyleTTS2 vocab")
    subparsers.add_parser("verify", help="Verify data integrity")
    
    args = parser.parse_args()
    
    commands = {
        "prepare": cmd_prepare,
        "extract-audio": cmd_extract_audio,
        "convert-weights": lambda: cmd_convert_weights(force=args.force),
        "patch-styletts2": cmd_patch_styletts2,
        "verify": cmd_verify,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()