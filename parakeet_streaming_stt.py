"""
parakeet_streaming_stt.py
=========================

CLI script to run NVIDIA Parakeet RNNT in:
- baseline (full-file) mode
- simulated streaming mode with overlapping chunks and latency stats

Usage examples
--------------

    # 1) Just run with defaults (downloads a sample wav)
    python parakeet_streaming_stt.py

    # 2) Use your own audio
    python parakeet_streaming_stt.py --audio path/to/audio.wav

    # 3) Change model and chunking
    python parakeet_streaming_stt.py \\
        --model nvidia/parakeet-rnnt-0.6b \\
        --chunk-dur 1.0 --overlap 0.25

Dependencies
------------
- nemo_toolkit[asr]
- torch
- numpy
- soundfile
- librosa
"""

import argparse
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr


SAMPLE_URL = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
DEFAULT_SAMPLE_RATE = 16000


@dataclass
class LatencySummary:
    full_text: str
    chunks: List[Dict[str, Any]]
    n_chunks: int
    total_latency_ms: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    rtf: float
    audio_duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Streaming STT + latency benchmarks for NVIDIA Parakeet RNNT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/parakeet-rnnt-1.1b",
        help="NeMo ASR model name.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file. If omitted, a sample file is downloaded.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Target sample rate for audio (Hz).",
    )
    parser.add_argument(
        "--chunk-dur",
        type=float,
        default=2.0,
        help="Chunk duration for streaming (seconds).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap between chunks (seconds).",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip multi-chunk-size benchmark; only run baseline + one streaming config.",
    )
    return parser.parse_args()


def device_string() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str) -> nemo_asr.models.EncDecRNNTBPEModel:
    print(f"Loading {model_name} on {device_string()} ...")
    t0 = time.time()
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
    model = model.to(device_string())
    model.eval()
    model.freeze()
    dt = time.time() - t0

    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded in {dt:.1f}s, params: {n_params_m:.0f}M")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory used: {mem:.2f} GB")

    return model


def ensure_audio(path: str | None, target_sr: int) -> Tuple[np.ndarray, int, str]:
    if path is None:
        local = "sample_audio.wav"
        if not os.path.exists(local):
            print(f"Downloading sample audio to {local} ...")
            urllib.request.urlretrieve(SAMPLE_URL, local)
        path = local

    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    duration = len(audio) / target_sr
    print(f"Audio file    : {path}")
    print(f"Sample rate   : {target_sr} Hz")
    print(f"Duration      : {duration:.2f}s")
    print(f"Samples       : {len(audio):,}")
    print(f"Dtype         : {audio.dtype}")
    print(f"Range         : [{audio.min():.3f}, {audio.max():.3f}]")
    return audio.astype(np.float32), target_sr, path


def extract_text_from_transcribe(output: Any) -> str:
    """
    Handle different NeMo RNNT transcribe() return types:
    - list of Hypotheses (with .text)
    - list of strings
    - tuple (greedy, beam) -> use greedy[0]
    """
    if isinstance(output, tuple) and len(output) > 0:
        output = output[0]

    if not isinstance(output, (list, tuple)) or len(output) == 0:
        return str(output)

    first = output[0]
    if isinstance(first, list) and first:
        first = first[0]

    if hasattr(first, "text"):
        return first.text
    return str(first)


def warmup(model: nemo_asr.models.EncDecRNNTBPEModel, sample_rate: int) -> None:
    if device_string() != "cuda":
        return
    print("Warming up CUDA kernels...")
    warmup_audio = np.zeros(sample_rate, dtype=np.float32)
    tmp = "/tmp/_parakeet_warmup.wav"
    sf.write(tmp, warmup_audio, sample_rate)
    _ = model.transcribe([tmp])
    torch.cuda.synchronize()
    print("Warmup done.\n")


def baseline_transcribe(
    model: nemo_asr.models.EncDecRNNTBPEModel,
    audio_path: str,
    duration_s: float,
) -> Tuple[str, float, float]:
    if device_string() == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = model.transcribe([audio_path])
    if device_string() == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    text = extract_text_from_transcribe(output)
    latency_ms = (t1 - t0) * 1000.0
    rtf = (t1 - t0) / duration_s
    return text, latency_ms, rtf


def streaming_transcribe(
    audio_array: np.ndarray,
    model: nemo_asr.models.EncDecRNNTBPEModel,
    sample_rate: int,
    chunk_duration_s: float,
    overlap_duration_s: float,
) -> LatencySummary:
    chunk_samples = int(chunk_duration_s * sample_rate)
    overlap_samples = int(overlap_duration_s * sample_rate)
    step_samples = chunk_samples - overlap_samples
    total_samples = len(audio_array)

    chunks_results: List[Dict[str, Any]] = []
    all_texts: List[str] = []
    latencies: List[float] = []

    n_chunks_est = (total_samples - overlap_samples) // step_samples + 1
    print(f"Chunk size     : {chunk_duration_s:.2f}s ({chunk_samples} samples)")
    print(f"Overlap        : {overlap_duration_s:.2f}s ({overlap_samples} samples)")
    print(f"Step           : {chunk_duration_s - overlap_duration_s:.2f}s ({step_samples} samples)")
    print(f"Expected chunks: ~{n_chunks_est}")
    print(f"Audio duration : {total_samples / sample_rate:.2f}s")
    print("-" * 70)

    total_t0 = time.perf_counter()
    chunk_idx = 0
    pos = 0

    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk = audio_array[pos:end]

        if len(chunk) < int(0.1 * sample_rate):
            break

        tmp_path = f"/tmp/_parakeet_chunk_{chunk_idx}.wav"
        sf.write(tmp_path, chunk, sample_rate)

        if device_string() == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = model.transcribe([tmp_path])
        if device_string() == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        text = extract_text_from_transcribe(result).strip()
        latency_ms = (t1 - t0) * 1000.0
        chunk_dur = len(chunk) / sample_rate

        chunk_info = {
            "idx": chunk_idx,
            "start_s": pos / sample_rate,
            "end_s": end / sample_rate,
            "duration_s": chunk_dur,
            "text": text,
            "latency_ms": latency_ms,
            "rtf": (t1 - t0) / chunk_dur,
        }
        chunks_results.append(chunk_info)
        latencies.append(latency_ms)

        print(
            f"[{chunk_idx:03d}] "
            f"{pos / sample_rate:6.2f}s-{end / sample_rate:6.2f}s  "
            f"latency={latency_ms:6.0f}ms  "
            f"RTFx={1.0 / chunk_info['rtf']:5.0f}x  "
            f"| {text}"
        )

        try:
            os.remove(tmp_path)
        except OSError:
            pass

        pos += step_samples
        chunk_idx += 1

    total_t1 = time.perf_counter()
    total_latency = (total_t1 - total_t0) * 1000.0
    audio_duration = total_samples / sample_rate

    lat_np = np.array(latencies, dtype=np.float32)
    summary = LatencySummary(
        full_text=" ".join(t for t in all_texts if t),
        chunks=chunks_results,
        n_chunks=len(chunks_results),
        total_latency_ms=float(total_latency),
        avg_latency_ms=float(lat_np.mean()) if len(lat_np) else 0.0,
        p50_latency_ms=float(np.percentile(lat_np, 50)) if len(lat_np) else 0.0,
        p95_latency_ms=float(np.percentile(lat_np, 95)) if len(lat_np) else 0.0,
        p99_latency_ms=float(np.percentile(lat_np, 99)) if len(lat_np) else 0.0,
        min_latency_ms=float(lat_np.min()) if len(lat_np) else 0.0,
        max_latency_ms=float(lat_np.max()) if len(lat_np) else 0.0,
        rtf=total_latency / 1000.0 / audio_duration if audio_duration > 0 else 0.0,
        audio_duration_s=audio_duration,
    )
    return summary


def main() -> None:
    args = parse_args()
    dev = device_string()
    print(f"Device: {dev}")

    model = load_model(args.model)
    audio, sr, audio_path = ensure_audio(args.audio, args.sample_rate)

    warmup(model, sr)

    print("\n=== Baseline (full-file) transcription ===")
    baseline_text, baseline_latency_ms, baseline_rtf = baseline_transcribe(
        model, audio_path, len(audio) / sr
    )
    print(f'Text    : "{baseline_text}"')
    print(f"Latency : {baseline_latency_ms:.0f} ms")
    print(f"RTF     : {baseline_rtf:.4f}  (Real-Time Factor, <1.0 = faster than real-time)")
    print(f"RTFx    : {1.0 / baseline_rtf:.0f}x\n")

    print("=== Streaming transcription (single config) ===")
    summary = streaming_transcribe(
        audio_array=audio,
        model=model,
        sample_rate=sr,
        chunk_duration_s=args.chunk_dur,
        overlap_duration_s=args.overlap,
    )

    print("\n" + "=" * 70)
    print("STREAMING RESULT:")
    print(f'  "{summary.full_text}"')
    print("\nLATENCY STATS:")
    print(f"  avg    = {summary.avg_latency_ms:.0f} ms")
    print(f"  p50    = {summary.p50_latency_ms:.0f} ms")
    print(f"  p95    = {summary.p95_latency_ms:.0f} ms")
    print(f"  min    = {summary.min_latency_ms:.0f} ms")
    print(f"  max    = {summary.max_latency_ms:.0f} ms")
    print(f"  total  = {summary.total_latency_ms:.0f} ms")
    print(f"  RTFx   = {1.0 / summary.rtf:.0f}x faster than real-time")
    print("=" * 70)


if __name__ == "__main__":
    main()

