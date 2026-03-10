#!/usr/bin/env python3
"""
Test client for the streaming STT WebSocket server.

Sends a WAV file as binary PCM frames to /ws/stt using the new protocol.

Usage:
    python scripts/test_streaming_client.py audio.wav
    python scripts/test_streaming_client.py audio.wav --url wss://your-ngrok/ws/stt
    python scripts/test_streaming_client.py audio.wav --vad-engine energy
    python scripts/test_streaming_client.py audio.wav --vad-engine none
"""

import argparse
import asyncio
import json
import sys
import time

import numpy as np

try:
    import librosa
    import websockets
except ImportError:
    print("Install: pip install librosa websockets")
    sys.exit(1)


async def run(
    audio_path: str,
    ws_url: str = "ws://localhost:8000/ws/stt",
    chunk_ms: int = 100,
    vad_engine: str = "silero",
    silence_duration_ms: int = 800,
):
    """Stream audio file to STT server and display results."""
    print(f"Loading: {audio_path}")
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    pcm = (audio * 32767).astype(np.int16)
    duration = len(pcm) / 16000
    print(f"Audio:   {duration:.1f}s ({len(pcm)} samples)")
    print(f"Server:  {ws_url}")
    print(f"VAD:     {vad_engine} (silence={silence_duration_ms}ms)")
    print()

    chunk_size = int(16000 * chunk_ms / 1000)
    # Align to 512 samples for Silero VAD compatibility
    chunk_size = max(512, (chunk_size // 512) * 512)

    latencies = []
    segment_count = 0
    import urllib.parse
    parsed_url = urllib.parse.urlparse(ws_url)
    host = parsed_url.netloc

    # ngrok free tier requires this header to bypass browser warning.
    # Uvicorn also requires Origin to match Host for WebSocket upgrades.
    extra_headers = {
        "ngrok-skip-browser-warning": "true",
        "Origin": f"https://{host}"
    }

    async with websockets.connect(ws_url, additional_headers=extra_headers) as ws:
        # Step 1: Send config
        config_msg = {
            "type": "config",
            "config": {
                "sample_rate": 16000,
                "encoding": "pcm_s16le",
                "vad": {
                    "enabled": vad_engine != "none",
                    "engine": vad_engine,
                    "silence_duration_ms": silence_duration_ms,
                },
            },
        }
        await ws.send(json.dumps(config_msg))

        # Step 2: Receive session.created
        session_msg = await ws.recv()
        session = json.loads(session_msg)
        if session.get("type") == "error":
            print(f"ERROR: {session['message']}")
            return
        print(f"Session: {session.get('session_id', '?')}")
        print("=" * 60)

        # Step 3: Stream binary PCM frames
        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i: i + chunk_size]
            if len(chunk) < 512:
                break

            # Send as raw bytes
            await ws.send(chunk.tobytes())

            # Check for any transcript events (non-blocking)
            while True:
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=0.02)
                    data = json.loads(resp)
                    if data.get("type", "").startswith("transcript."):
                        segment_count += 1
                        lat = data.get("latency_ms", 0)
                        latencies.append(lat)
                        marker = "✓" if "final" in data["type"] else "…"
                        print(
                            f"  [{marker}] seg={data.get('segment_id', '?'):>2}  "
                            f"latency={lat:>6.0f}ms  "
                            f"| {data.get('text', '')}"
                        )
                    elif data.get("type") == "error":
                        print(f"  [ERROR] {data.get('message', '')}")
                except asyncio.TimeoutError:
                    break

            # Simulate real-time playback speed
            await asyncio.sleep(chunk_ms / 1000.0)

        # Step 4: Send end-of-stream
        await ws.send(json.dumps({"type": "end"}))

        # Step 5: Receive final events
        while True:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(resp)

                if data.get("type", "").startswith("transcript."):
                    segment_count += 1
                    lat = data.get("latency_ms", 0)
                    latencies.append(lat)
                    marker = "✓" if "final" in data["type"] else "…"
                    print(
                        f"  [{marker}] seg={data.get('segment_id', '?'):>2}  "
                        f"latency={lat:>6.0f}ms  "
                        f"| {data.get('text', '')}"
                    )
                elif data.get("type") == "session.ended":
                    print()
                    print("=" * 60)
                    print(f"Session ended:")
                    print(f"  Segments:    {data.get('segments_transcribed', 0)}")
                    print(f"  Audio:       {data.get('total_audio_s', 0):.1f}s")
                    print(f"  Session:     {data.get('session_duration_s', 0):.1f}s")
                    break
                elif data.get("type") == "error":
                    print(f"  [ERROR] {data.get('message', '')}")
                    break
            except asyncio.TimeoutError:
                print("  [TIMEOUT] No response after 10s")
                break

    # Latency stats
    if latencies:
        latencies_np = np.array(latencies)
        print(f"\nLatency Stats ({len(latencies)} segments):")
        print(f"  avg={np.mean(latencies_np):.0f}ms  "
              f"p50={np.percentile(latencies_np, 50):.0f}ms  "
              f"p95={np.percentile(latencies_np, 95):.0f}ms  "
              f"min={np.min(latencies_np):.0f}ms  "
              f"max={np.max(latencies_np):.0f}ms")

    total_time = time.time() - stream_start
    print(f"  Total wall time: {total_time:.1f}s (for {duration:.1f}s audio)")
    print(f"  Real-time factor: {total_time / duration:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test STT streaming client")
    parser.add_argument("audio", help="Path to WAV file")
    parser.add_argument("--url", default="ws://localhost:8000/ws/stt")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Chunk size in ms")
    parser.add_argument("--vad-engine", choices=["silero", "energy", "none"], default="silero")
    parser.add_argument("--silence-ms", type=int, default=800, help="Silence duration for VAD")
    args = parser.parse_args()

    asyncio.run(run(
        args.audio, args.url, args.chunk_ms,
        args.vad_engine, args.silence_ms,
    ))
