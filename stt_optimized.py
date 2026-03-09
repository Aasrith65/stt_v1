"""
STT Optimizations: VAD, TTFW, FP16

- VAD (Silero): Skip silence, only transcribe speech
- Variable first chunk: Shorter first chunk for faster time-to-first-word
- FP16: Half precision for ~1.5x faster inference on GPU
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch


def load_silero_vad():
    """Load Silero VAD via torch.hub (no extra install)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils


def get_speech_segments(
    audio: np.ndarray,
    sample_rate: int,
    vad_model,
    get_speech_timestamps,
    min_speech_duration_ms: float = 250,
    min_silence_duration_ms: float = 100,
    speech_pad_ms: float = 30,
    threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Return list of (start_sample, end_sample) for speech segments.
    Skips silence.
    """
    if get_speech_timestamps is None:
        return [(0, len(audio))]  # No VAD: treat whole audio as speech

    timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        threshold=threshold,
    )
    if not timestamps:
        return []
    return [(t["start"], t["end"]) for t in timestamps]


def extract_text(output: Any) -> str:
    """Handle NeMo transcribe output."""
    if isinstance(output, tuple) and len(output) > 0:
        output = output[0]
    if not isinstance(output, (list, tuple)) or len(output) == 0:
        return str(output)
    first = output[0]
    if isinstance(first, list) and first:
        first = first[0]
    return first.text if hasattr(first, "text") else str(first)


def streaming_transcribe_optimized(
    audio_array: np.ndarray,
    model,
    sample_rate: int,
    *,
    first_chunk_s: float = 1.0,
    rest_chunk_s: float = 2.0,
    overlap_s: float = 0.25,
    use_vad: bool = True,
    use_fp16: bool = False,
    vad_model=None,
    get_speech_timestamps=None,
    extract_text_fn: Optional[Callable] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[str, List[float], float]:
    """
    Optimized streaming transcription.

    - first_chunk_s: Shorter first chunk → faster TTFW (~1.1s vs ~2.1s)
    - use_vad: Skip silence segments (saves compute, cleaner output)
    - use_fp16: Half precision (default False; Parakeet may have accuracy issues)

    Returns: (full_text, latencies_list, ttfw_ms)
    """
    extract = extract_text_fn or extract_text
    texts: List[str] = []
    latencies: List[float] = []
    ttfw_ms: float = -1.0

    if use_fp16 and device == "cuda":
        model = model.half()

    # Get speech segments (or full audio if no VAD)
    if use_vad and vad_model is not None and get_speech_timestamps is not None:
        segments = get_speech_segments(
            audio_array, sample_rate, vad_model, get_speech_timestamps
        )
        if not segments:
            return "", [], 0.0
    else:
        segments = [(0, len(audio_array))]

    for seg_start, seg_end in segments:
        seg_audio = audio_array[seg_start:seg_end].astype(np.float32)
        seg_len = len(seg_audio)
        if seg_len < int(0.1 * sample_rate):
            continue

        # Variable chunk size: first chunk shorter for TTFW
        pos = 0
        chunk_idx = 0
        while pos < seg_len:
            chunk_s = first_chunk_s if chunk_idx == 0 else rest_chunk_s
            chunk_n = int(chunk_s * sample_rate)
            overlap_n = int(overlap_s * sample_rate)
            step_n = max(1, chunk_n - overlap_n)

            end = min(pos + chunk_n, seg_len)
            chunk = seg_audio[pos:end]
            if len(chunk) < int(0.1 * sample_rate):
                break

            tmp = f"/tmp/_opt_chunk_{chunk_idx}.wav"
            sf.write(tmp, chunk, sample_rate)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model.transcribe([tmp], batch_size=1, verbose=False)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000
            latencies.append(lat_ms)
            if ttfw_ms < 0:
                ttfw_ms = (pos / sample_rate) * 1000 + lat_ms

            text = extract(result).strip()
            texts.append(text)
            if verbose:
                abs_start = (seg_start + pos) / sample_rate
                abs_end = (seg_start + end) / sample_rate
                print(f"[{chunk_idx:02d}] {abs_start:.2f}s-{abs_end:.2f}s  {lat_ms:.0f}ms  | {text}")

            try:
                os.remove(tmp)
            except OSError:
                pass
            pos += step_n
            chunk_idx += 1

    if use_fp16 and device == "cuda":
        model = model.float()

    return " ".join(texts), latencies, ttfw_ms
#test