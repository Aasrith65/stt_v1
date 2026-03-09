"""
Continuous Streaming Speech-to-Text with Whisper Base
=====================================================

Listens to the microphone continuously, detects speech via energy-based VAD,
transcribes each utterance with Whisper, and prints results with latency stats.

Usage:
    python stream.py
    python stream.py --language english
    python stream.py --chunk-size 2 --silence-threshold 0.01
    python stream.py --device cpu

Press Ctrl+C to stop.
"""

import argparse
import collections
import sys
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_ID = "openai/whisper-base"
SAMPLE_RATE = 16_000  # Whisper expects 16 kHz


# ── Model Loading ────────────────────────────────────────────────────────────
def load_model(device: Optional[str] = None):
    """Load Whisper processor and model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model '{MODEL_ID}' on {device} ...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    print(f"Model loaded on {device}\n")
    return processor, model, device


# ── Transcription ────────────────────────────────────────────────────────────
def transcribe_chunk(
    audio: np.ndarray,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    language: Optional[str] = None,
    task: str = "transcribe",
) -> str:
    """Transcribe a single audio chunk and return the text."""
    if language:
        forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        model.config.forced_decoder_ids = forced_ids
    else:
        model.config.forced_decoder_ids = None

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return text


# ── VAD (Voice Activity Detection) ──────────────────────────────────────────
def compute_rms(audio: np.ndarray) -> float:
    """Compute root-mean-square energy of audio."""
    return float(np.sqrt(np.mean(audio ** 2)))


# ── Streaming Loop ───────────────────────────────────────────────────────────
def stream_transcribe(
    processor,
    model,
    device: str,
    language: Optional[str] = None,
    task: str = "transcribe",
    chunk_duration: float = 2.0,
    silence_threshold: float = 0.01,
    silence_duration: float = 0.8,
    min_speech_duration: float = 0.5,
    max_speech_duration: float = 30.0,
):
    """
    Continuously listen to the microphone and transcribe speech.

    How it works:
      1. Audio is captured in small frames (~100ms) via a callback.
      2. Energy-based VAD detects when speech starts (RMS > threshold).
      3. Audio is buffered while speech is ongoing.
      4. When silence is detected (RMS < threshold for silence_duration),
         the buffered audio is sent to Whisper for transcription.
      5. Latency is measured from end-of-speech to transcription-complete.

    Args:
        chunk_duration:      Duration of each analysis frame (seconds).
        silence_threshold:   RMS energy below this = silence.
        silence_duration:    Seconds of consecutive silence to end an utterance.
        min_speech_duration: Ignore utterances shorter than this (seconds).
        max_speech_duration: Force-transcribe if speech exceeds this (seconds).
    """
    FRAME_DURATION = 0.1  # 100ms frames for responsive VAD
    frame_samples = int(FRAME_DURATION * SAMPLE_RATE)

    # Shared state between callback and main thread
    audio_queue = collections.deque()
    lock = threading.Lock()

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio frame."""
        if status:
            print(f"  [audio warning: {status}]", file=sys.stderr)
        with lock:
            audio_queue.append(indata[:, 0].copy())  # mono

    # ── Latency tracking ─────────────────────────────────────────────────
    latencies = []

    def print_stats():
        if not latencies:
            return
        avg = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        mn = np.min(latencies)
        mx = np.max(latencies)
        print(f"  [latency] avg={avg:.0f}ms  p50={p50:.0f}ms  "
              f"p95={p95:.0f}ms  min={mn:.0f}ms  max={mx:.0f}ms  "
              f"n={len(latencies)}")

    # ── State machine ────────────────────────────────────────────────────
    speech_buffer = []      # accumulated speech frames
    is_speaking = False
    silence_counter = 0.0   # seconds of consecutive silence
    utterance_count = 0
    speech_start_time = 0.0

    print("=" * 60)
    print("  STREAMING SPEECH-TO-TEXT")
    print("  Speak into your microphone. Press Ctrl+C to stop.")
    print(f"  Silence threshold: {silence_threshold}  |  "
          f"Silence gap: {silence_duration}s")
    print("=" * 60)
    print()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=audio_callback,
        ):
            while True:
                time.sleep(FRAME_DURATION)

                # Drain queue
                frames = []
                with lock:
                    while audio_queue:
                        frames.append(audio_queue.popleft())

                if not frames:
                    continue

                for frame in frames:
                    rms = compute_rms(frame)

                    if not is_speaking:
                        # ── Waiting for speech ───────────────────────
                        if rms > silence_threshold:
                            is_speaking = True
                            speech_buffer = [frame]
                            silence_counter = 0.0
                            speech_start_time = time.time()
                            print(">> listening...", end="", flush=True)
                    else:
                        # ── Currently in speech ──────────────────────
                        speech_buffer.append(frame)

                        if rms < silence_threshold:
                            silence_counter += FRAME_DURATION
                        else:
                            silence_counter = 0.0

                        speech_duration = time.time() - speech_start_time

                        # End utterance on silence or max duration
                        should_end = (
                            silence_counter >= silence_duration
                            or speech_duration >= max_speech_duration
                        )

                        if should_end:
                            end_of_speech_time = time.time()
                            audio_data = np.concatenate(speech_buffer)
                            duration = len(audio_data) / SAMPLE_RATE

                            if duration < min_speech_duration:
                                # Too short — probably noise, discard
                                print(" (too short, skipped)")
                                is_speaking = False
                                speech_buffer = []
                                silence_counter = 0.0
                                continue

                            # Transcribe
                            t0 = time.perf_counter()
                            text = transcribe_chunk(
                                audio_data, processor, model,
                                device, language, task,
                            )
                            t1 = time.perf_counter()

                            inference_ms = (t1 - t0) * 1000
                            e2e_ms = (time.time() - end_of_speech_time) * 1000
                            latencies.append(e2e_ms)
                            utterance_count += 1

                            # Print result
                            print(f"\r[{utterance_count:03d}] "
                                  f"({duration:.1f}s) "
                                  f"latency={e2e_ms:.0f}ms "
                                  f"inference={inference_ms:.0f}ms")
                            print(f"  >> {text}")
                            print()

                            # Reset state
                            is_speaking = False
                            speech_buffer = []
                            silence_counter = 0.0

    except KeyboardInterrupt:
        print("\n\nStopping...\n")

    # ── Final stats ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  SESSION SUMMARY: {utterance_count} utterances transcribed")
    print_stats()
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Continuous streaming STT with Whisper Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stream.py
  python stream.py --language english
  python stream.py --silence-threshold 0.015 --silence-duration 1.0
  python stream.py --task translate --language french
        """,
    )
    parser.add_argument(
        "--language", "-l", type=str, default=None,
        help="Language of the audio (e.g. english, french). Auto-detect if omitted.",
    )
    parser.add_argument(
        "--task", "-t", type=str, choices=["transcribe", "translate"],
        default="transcribe",
        help="'transcribe' or 'translate' to English (default: transcribe)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu' or 'cuda' (auto-detected)",
    )
    parser.add_argument(
        "--silence-threshold", type=float, default=0.01,
        help="RMS energy threshold for silence detection (default: 0.01)",
    )
    parser.add_argument(
        "--silence-duration", type=float, default=0.8,
        help="Seconds of silence to end an utterance (default: 0.8)",
    )
    parser.add_argument(
        "--min-speech", type=float, default=0.5,
        help="Minimum speech duration in seconds to transcribe (default: 0.5)",
    )
    parser.add_argument(
        "--max-speech", type=float, default=30.0,
        help="Maximum speech duration before forced transcription (default: 30)",
    )

    args = parser.parse_args()
    processor, model, device = load_model(args.device)

    stream_transcribe(
        processor, model, device,
        language=args.language,
        task=args.task,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        min_speech_duration=args.min_speech,
        max_speech_duration=args.max_speech,
    )


if __name__ == "__main__":
    main()
