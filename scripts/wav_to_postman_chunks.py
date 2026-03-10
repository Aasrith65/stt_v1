#!/usr/bin/env python3
"""
Convert a WAV file to base64 chunks for Postman WebSocket testing.

Usage: python scripts/wav_to_postman_chunks.py audio.wav [--chunk-ms 200]

Outputs JSON with chunks you can copy into Postman:
  {"audio": "<chunk1_base64>"}  → send
  {"audio": "<chunk2_base64>"}  → send
  ...
  {"end": true}                 → send when done
"""

import argparse
import base64
import json
import sys

import numpy as np

try:
    import librosa
except ImportError:
    print("pip install librosa")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="Path to WAV file")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Chunk size in ms")
    parser.add_argument("--output", "-o", help="Write chunks to JSON file")
    args = parser.parse_args()

    audio, _ = librosa.load(args.wav, sr=16000, mono=True)
    pcm = (audio * 32767).astype(np.int16)
    chunk_samples = int(16000 * args.chunk_ms / 1000)
    chunk_samples = max(512, (chunk_samples // 512) * 512)

    chunks = []
    for i in range(0, len(pcm), chunk_samples):
        chunk = pcm[i : i + chunk_samples]
        if len(chunk) < 512:
            break
        b64 = base64.b64encode(chunk.tobytes()).decode()
        chunks.append({"audio": b64})

    chunks.append({"end": True})

    if args.output:
        with open(args.output, "w") as f:
            json.dump(chunks, f, indent=2)
        print(f"Wrote {len(chunks)} messages to {args.output}")
    else:
        for i, msg in enumerate(chunks):
            print(f"--- Message {i + 1} ---")
            print(json.dumps(msg))
            print()


if __name__ == "__main__":
    main()
