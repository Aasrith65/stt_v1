#!/usr/bin/env python3
"""
Test client for streaming STT WebSocket server.

Sends a WAV file in chunks to simulate streaming.
Usage: python scripts/test_streaming_client.py path/to/audio.wav
"""

import argparse
import asyncio
import base64
import json
import sys

import numpy as np

try:
    import librosa
    import websockets
except ImportError:
    print("Install: pip install librosa websockets")
    sys.exit(1)


async def run(audio_path: str, ws_url: str = "ws://localhost:8000/ws/transcribe", chunk_ms: int = 100):
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    pcm = (audio * 32767).astype(np.int16)
    chunk_size = int(16000 * chunk_ms / 1000)
    chunk_size = (chunk_size // 512) * 512  # align to VAD window

    async with websockets.connect(ws_url) as ws:
        msg = await ws.recv()
        print(f"Server: {msg}")

        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i : i + chunk_size]
            if len(chunk) < 512:
                break
            b64 = base64.b64encode(chunk.tobytes()).decode()
            await ws.send(json.dumps({"audio": b64}))
            while True:
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    data = json.loads(resp)
                    if "text" in data:
                        print(f"  → {data['text']}")
                    if "error" in data:
                        print(f"  Error: {data['error']}")
                except asyncio.TimeoutError:
                    break

        await ws.send(json.dumps({"end": True}))
        while True:
            resp = await ws.recv()
            data = json.loads(resp)
            if "text" in data:
                print(f"  → {data['text']}")
            if data.get("is_final") or "error" in data:
                break

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to WAV file")
    parser.add_argument("--url", default="ws://localhost:8000/ws/transcribe")
    parser.add_argument("--chunk-ms", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(run(args.audio, args.url, args.chunk_ms))
