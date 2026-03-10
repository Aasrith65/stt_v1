#!/usr/bin/env python3
"""
Mock LLM consumer for the STT streaming service.

Connects as a WebSocket client, streams audio, and demonstrates how
an LLM orchestrator would consume transcript events in real-time.

Usage:
    python scripts/test_llm_consumer.py audio.wav
    python scripts/test_llm_consumer.py audio.wav --url wss://your-ngrok/ws/stt
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Optional

import numpy as np

try:
    import librosa
    import websockets
except ImportError:
    print("Install: pip install librosa websockets")
    sys.exit(1)


class MockLLMProcessor:
    """
    Simulates how an LLM agent would consume transcript events.

    In production, this would be replaced with:
    - OpenAI API call with the accumulated transcript
    - Gemini / Claude API call
    - Local LLM inference
    - Agent framework (LangChain, AutoGen, etc.)
    """

    def __init__(self):
        self.transcript_buffer = []
        self.full_text = ""

    def on_transcript(self, event: dict):
        """Called for each transcript.final event from STT."""
        text = event.get("text", "")
        segment_id = event.get("segment_id", 0)
        timestamp_ms = event.get("timestamp_ms", 0)

        self.transcript_buffer.append(text)
        self.full_text = " ".join(self.transcript_buffer)

        print(f"\n  ╔══ LLM Input (segment {segment_id}) ══")
        print(f"  ║ New text:       \"{text}\"")
        print(f"  ║ Accumulated:    \"{self.full_text}\"")
        print(f"  ║ Timestamp:      {timestamp_ms:.0f}ms")
        print(f"  ║")

        # Simulate LLM processing decision
        if len(self.full_text.split()) >= 5:
            t0 = time.perf_counter()
            response = self._mock_llm_call(self.full_text)
            llm_latency = (time.perf_counter() - t0) * 1000
            print(f"  ║ LLM Response:   \"{response}\"")
            print(f"  ║ LLM Latency:    {llm_latency:.0f}ms")
        else:
            print(f"  ║ (Waiting for more context before LLM call)")

        print(f"  ╚{'═' * 40}")

    def _mock_llm_call(self, text: str) -> str:
        """Simulated LLM call. Replace with actual API in production."""
        # Simulate ~50ms latency
        time.sleep(0.05)
        word_count = len(text.split())
        return f"[Mock LLM] Received {word_count} words. Acknowledged."

    def get_summary(self) -> dict:
        return {
            "total_segments": len(self.transcript_buffer),
            "total_words": len(self.full_text.split()),
            "full_transcript": self.full_text,
        }


async def run(
    audio_path: str,
    ws_url: str = "ws://localhost:8000/ws/stt",
    origin: Optional[str] = None,
    use_ngrok_header: bool = False,
):
    print("=" * 60)
    print("  MOCK LLM CONSUMER — STT Integration Demo")
    print("=" * 60)

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    pcm = (audio * 32767).astype(np.int16)
    print(f"\nAudio: {audio_path} ({len(pcm) / 16000:.1f}s)")
    print(f"Server: {ws_url}\n")

    llm = MockLLMProcessor()
    chunk_ms = 100
    chunk_size = max(512, (int(16000 * chunk_ms / 1000) // 512) * 512)

    connect_kwargs = {}
    if origin:
        connect_kwargs["origin"] = origin
    if use_ngrok_header:
        connect_kwargs["additional_headers"] = {
            "ngrok-skip-browser-warning": "true",
        }

    async with websockets.connect(ws_url, **connect_kwargs) as ws:
        # Send config
        await ws.send(json.dumps({
            "type": "config",
            "config": {
                "sample_rate": 16000,
                "encoding": "pcm_s16le",
                "vad": {"enabled": True, "engine": "silero"},
            },
        }))

        # Wait for session.created
        session = json.loads(await ws.recv())
        print(f"Session: {session.get('session_id', '?')}")
        print("-" * 60)

        # Stream audio
        for i in range(0, len(pcm), chunk_size):
            chunk = pcm[i: i + chunk_size]
            if len(chunk) < 512:
                break
            await ws.send(chunk.tobytes())

            # Check for events
            while True:
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=0.02)
                    data = json.loads(resp)
                    if data.get("type") == "transcript.final":
                        llm.on_transcript(data)
                    elif data.get("type") == "error":
                        print(f"  [ERROR] {data.get('message')}")
                except asyncio.TimeoutError:
                    break

            await asyncio.sleep(chunk_ms / 1000.0)

        # End stream
        await ws.send(json.dumps({"type": "end"}))

        # Collect remaining events
        while True:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(resp)
                if data.get("type") == "transcript.final":
                    llm.on_transcript(data)
                elif data.get("type") == "session.ended":
                    break
            except asyncio.TimeoutError:
                break

    # Summary
    summary = llm.get_summary()
    print()
    print("=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Segments processed: {summary['total_segments']}")
    print(f"  Total words:        {summary['total_words']}")
    print(f"  Full transcript:")
    print(f"    \"{summary['full_transcript']}\"")
    print("=" * 60)
    print()
    print("In production, replace MockLLMProcessor._mock_llm_call()")
    print("with your actual LLM API call (OpenAI, Gemini, etc.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock LLM consumer for STT")
    parser.add_argument("audio", help="Path to WAV file")
    parser.add_argument("--url", default="ws://localhost:8000/ws/stt")
    parser.add_argument("--origin", help="Optional Origin header for strict proxy deployments")
    parser.add_argument("--ngrok", action="store_true", help="Send ngrok browser-warning bypass header")
    args = parser.parse_args()
    asyncio.run(run(args.audio, args.url, args.origin, args.ngrok))
