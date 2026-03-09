"""
Local Speech-to-Text using OpenAI Whisper Base
https://huggingface.co/openai/whisper-base

Supports:
  - Transcribing audio files (WAV, MP3, FLAC, etc.)
  - Recording from microphone and transcribing
  - Multiple languages (auto-detect or specify)
"""

import argparse
import sys
from typing import Optional

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "openai/whisper-base"
SAMPLE_RATE = 16_000  # Whisper expects 16 kHz audio


def load_model(device: Optional[str] = None):
    """Load the Whisper processor and model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f" Loading model '{MODEL_ID}' on {device} …")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    print(" Model loaded successfully!\n")
    return processor, model, device


def transcribe_audio(
    audio_array: np.ndarray,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    language: Optional[str] = None,
    task: str = "transcribe",
) -> str:
    """
    Transcribe a numpy audio array (mono, 16 kHz) to text.

    Args:
        audio_array: 1-D float32 numpy array of audio samples at 16 kHz.
        processor:   WhisperProcessor instance.
        model:       WhisperForConditionalGeneration instance.
        device:      "cpu" or "cuda".
        language:    Language code (e.g. "english", "french"). None = auto-detect.
        task:        "transcribe" or "translate" (translate → English).

    Returns:
        Transcribed text string.
    """
    # Build forced decoder IDs for language / task control
    if language:
        forced_ids = processor.get_decoder_prompt_ids(
            language=language, task=task
        )
        model.config.forced_decoder_ids = forced_ids
    else:
        model.config.forced_decoder_ids = None

    # Pre-process audio → log-Mel spectrogram features
    inputs = processor(
        audio_array,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    # Generate token IDs
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    # Decode tokens → text
    transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0].strip()

    return transcription


# ── File-based transcription ─────────────────────────────────────────────────
def transcribe_file(
    file_path: str,
    processor,
    model,
    device: str,
    language: Optional[str] = None,
    task: str = "transcribe",
) -> str:
    """Load an audio file and transcribe it."""
    import librosa

    print(f"🎧 Loading audio file: {file_path}")
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE
    print(f"   Duration: {duration:.1f}s | Samples: {len(audio)}")

    print("🔄 Transcribing …")
    text = transcribe_audio(audio, processor, model, device, language, task)
    return text

def transcribe_microphone(
    duration: float,
    processor,
    model,
    device: str,
    language: Optional[str] = None,
    task: str = "transcribe",
) -> str:
    """Record from the default microphone for `duration` seconds, then transcribe."""
    import sounddevice as sd

    print(f"🎙️  Recording for {duration:.1f} seconds … (speak now!)")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # block until recording finishes
    audio = audio.squeeze()  # (N, 1) → (N,)
    print("✅ Recording complete!\n")

    print("🔄 Transcribing …")
    text = transcribe_audio(audio, processor, model, device, language, task)
    return text


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Local Speech-to-Text with Whisper Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe an audio file
  python transcribe.py --file recording.wav

  # Record 10 seconds from mic and transcribe
  python transcribe.py --mic --duration 10

  # Translate French audio to English text
  python transcribe.py --file french.mp3 --language french --task translate

  # Force CPU even if GPU is available
  python transcribe.py --file audio.wav --device cpu
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", "-f", type=str, help="Path to an audio file to transcribe"
    )
    input_group.add_argument(
        "--mic", "-m", action="store_true", help="Record from microphone"
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=5.0,
        help="Recording duration in seconds (only used with --mic, default: 5)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language of the audio (e.g. english, french). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["transcribe", "translate"],
        default="transcribe",
        help="'transcribe' = same-language STT, 'translate' = translate to English (default: transcribe)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on: 'cpu' or 'cuda' (auto-detected if omitted)",
    )

    args = parser.parse_args()

    # Load model once
    processor, model, device = load_model(args.device)

    if args.file:
        text = transcribe_file(
            args.file, processor, model, device, args.language, args.task
        )
    else:
        text = transcribe_microphone(
            args.duration, processor, model, device, args.language, args.task
        )

    print("\n" + "=" * 60)
    print("📝 TRANSCRIPTION:")
    print("=" * 60)
    print(text)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
