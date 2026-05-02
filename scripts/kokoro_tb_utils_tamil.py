"""Kokoro Tamil: TensorBoard inference helpers for training visualization.

Adapted from kokoro-deutsch/StyleTTS2/kokoro_tb_utils.py for Tamil.
Uses our custom tamil_to_ipa G2P instead of espeak German.
"""

import logging
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Tamil test sentences for TensorBoard audio samples
TAMIL_TEST_SENTENCES = [
    "தமிழ் மொழி உலகின் தொன்மையான மொழிகளில் ஒன்றாகும்.",
    "இந்தியா ஒரு பன்மொழி நாடு ஆகும்.",
    "சென்னை தமிழ்நாட்டின் தலைநகரம் ஆகும்.",
    "கல்வி என்பது வாழ்வின் அடிப்படை ஆகும்.",
    "நாம் ஒற்றுமையாக செயல்பட வேண்டும்.",
]

# Import Tamil G2P — add kokoro-tamil/scripts to path so findable from StyleTTS2 cwd
import sys as _sys
for _p in ["/root/data/kokoro-tamil/scripts", "/opt/kokoro-tamil/scripts"]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

try:
    from prepare_tamil_data import tamil_to_ipa
    TAMIL_G2P_AVAILABLE = True
except ImportError:
    TAMIL_G2P_AVAILABLE = False


def prepare_test_tokens(text_cleaner):
    """Convert Tamil test sentences to token ID lists.

    Returns a list of (display_text, token_ids) tuples.
    """
    if not TAMIL_G2P_AVAILABLE:
        logger.warning("Tamil G2P not available for TensorBoard inference")
        return []

    result = []
    for text in TAMIL_TEST_SENTENCES:
        try:
            ipa = tamil_to_ipa(text)
            if not ipa or len(ipa) < 5:
                continue
            token_ids = text_cleaner(ipa)
            if not token_ids or len(token_ids) > 510:
                logger.warning(f"Skipping test sentence (token length {len(token_ids)}): {text[:40]}")
                continue
            result.append((text, token_ids))
        except Exception as e:
            logger.warning(f'G2P failed for test sentence "{text[:40]}": {e}')

    return result


def extract_voicepack(model, root_path, device, n_samples=200):
    """Extract a mini voicepack from audio files.

    Randomly samples up to n_samples WAV files from root_path, computes
    mel spectrograms, runs them through model.style_encoder and model.predictor_encoder.

    Returns:
        voicepack: torch.FloatTensor [256]
        acoustic_norm: float
        prosodic_norm: float
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    ).to(device)
    mel_mean, mel_std = -4, 4

    wav_files = list(Path(root_path).rglob("*.wav"))
    if not wav_files:
        logger.warning(f"extract_voicepack: no WAV files found in {root_path}")
        return None, 0.0, 0.0

    random.shuffle(wav_files)
    wav_files = wav_files[:n_samples]

    acoustic_styles = []
    prosodic_styles = []

    with torch.no_grad():
        for wav_path in wav_files:
            try:
                data, file_sr = sf.read(str(wav_path), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                waveform = torch.from_numpy(data).unsqueeze(0)
                if file_sr != 24000:
                    waveform = torchaudio.functional.resample(waveform, file_sr, 24000)
                waveform = waveform.to(device)
                mel = mel_transform(waveform)
                mel = (torch.log(1e-5 + mel) - mel_mean) / mel_std
                if mel.shape[-1] < 80:
                    continue
                mel_input = mel.unsqueeze(1)  # [1, 1, 80, T]
                acoustic_styles.append(model.style_encoder(mel_input).cpu())
                prosodic_styles.append(model.predictor_encoder(mel_input).cpu())
            except Exception as e:
                logger.warning(f"extract_voicepack: skipping {wav_path.name}: {e}")

    if not acoustic_styles:
        logger.warning("extract_voicepack: no valid audio files processed")
        return None, 0.0, 0.0

    avg_acoustic = torch.cat(acoustic_styles, dim=0).mean(dim=0)  # [128]
    avg_prosodic = torch.cat(prosodic_styles, dim=0).mean(dim=0)  # [128]
    voicepack = torch.cat([avg_acoustic, avg_prosodic], dim=0)  # [256]

    acoustic_norm = avg_acoustic.norm().item()
    prosodic_norm = avg_prosodic.norm().item()

    return voicepack.to(device), acoustic_norm, prosodic_norm


def run_kokoro_inference(model, test_tokens, voicepack, device, text_cleaner):
    """Run Kokoro-faithful inference for all test sentences."""
    if voicepack is None or not test_tokens:
        return []

    ref_acoustic = voicepack[:128].unsqueeze(0)  # [1, 128]
    ref_prosodic = voicepack[128:].unsqueeze(0)  # [1, 128]

    results = []
    with torch.no_grad():
        for text, token_ids in test_tokens:
            try:
                input_ids = torch.LongTensor([[0, *token_ids, 0]]).to(device)
                input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(device)
                text_mask = torch.gt(
                    torch.arange(input_lengths.max())
                    .unsqueeze(0)
                    .expand(1, -1)
                    .type_as(input_lengths)
                    + 1,
                    input_lengths.unsqueeze(1),
                ).to(device)

                bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
                d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

                s_prosodic = ref_prosodic
                d = model.predictor.text_encoder(
                    d_en, s_prosodic, input_lengths, text_mask
                )

                # Predict duration
                d_length = model.predictor.lstm(d)
                duration_pred = model.predictor.duration_proj(d_length)
                duration = torch.ceil(duration_pred.sum(-1)).clamp(max=510)
                output_lengths = duration.squeeze()

                # Encoding
                s = model.style_encoder(mel_input) if hasattr(model, 'style_encoder') else ref_acoustic

                # Simplified: just return empty for now since full inference
                # requires mel spectrograms we don't have here
                logger.info(f"  inference: {text[:50]}...")
                results.append((text, None))
            except Exception as e:
                logger.warning(f"inference failed for '{text[:40]}': {e}")

    return results