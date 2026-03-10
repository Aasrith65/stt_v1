"""
Streaming STT Pipeline — refactored from stt_streaming_server.py.

Combines a pluggable VAD engine with the Parakeet ASR model to produce
structured transcript events suitable for WebSocket streaming and LLM consumption.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

from config import SessionConfig, AudioEncoding
from vad import BaseVAD, VADEventType, create_vad


# ── Transcript events ────────────────────────────────────────────────────────

@dataclass
class TranscriptEvent:
    """Structured transcript event for downstream consumers (WebSocket, LLM, etc.)."""
    type: str                 # "transcript.partial" or "transcript.final"
    text: str
    segment_id: int
    timestamp_ms: float       # timestamp relative to session start
    audio_start_ms: float     # where this segment starts in the audio
    audio_end_ms: float       # where this segment ends in the audio
    latency_ms: float = 0.0   # inference latency
    confidence: float = 0.0   # placeholder for future confidence scores

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "text": self.text,
            "segment_id": self.segment_id,
            "timestamp_ms": round(self.timestamp_ms, 1),
            "audio_start_ms": round(self.audio_start_ms, 1),
            "audio_end_ms": round(self.audio_end_ms, 1),
            "latency_ms": round(self.latency_ms, 1),
            "confidence": round(self.confidence, 3),
        }


# ── Text extraction helper ──────────────────────────────────────────────────

def _extract_text(output) -> str:
    """Handle various NeMo transcribe output formats."""
    if isinstance(output, tuple) and len(output) > 0:
        output = output[0]
    if not isinstance(output, (list, tuple)) or len(output) == 0:
        return str(output)
    first = output[0]
    if isinstance(first, list) and first:
        first = first[0]
    return first.text if hasattr(first, "text") else str(first)


# ── Streaming STT Pipeline ──────────────────────────────────────────────────

SAMPLE_RATE = 16000


class StreamingPipeline:
    """
    Per-session streaming pipeline: audio → VAD → ASR → TranscriptEvents.

    Usage:
        pipeline = StreamingPipeline(asr_model, config, device)
        events = pipeline.feed_audio(pcm_bytes)  # binary PCM
        events = pipeline.flush()                  # end of stream
    """

    def __init__(
        self,
        asr_model,
        config: SessionConfig,
        device: str = "cuda",
    ):
        self.asr_model = asr_model
        self.config = config
        self.device = device
        self.sample_rate = config.sample_rate

        # Create VAD engine
        self.vad: BaseVAD = create_vad(config.vad, self.sample_rate)

        # Audio buffer: accumulates all audio for segment extraction
        self.audio_buffer = np.array([], dtype=np.float32)
        self._buffer_start_sample = 0  # Absolute sample offset of audio_buffer[0]
        self._segment_id = 0
        self._session_start = time.time()

        # Track speech boundaries from VAD
        self._speech_start_sample: Optional[int] = None
        self._pending_segments: List[tuple] = []  # Absolute [(start_sample, end_sample), ...]

    def feed_audio(self, data: bytes) -> List[TranscriptEvent]:
        """
        Feed raw PCM bytes. Returns any completed transcript events.
        Binary data is decoded according to session encoding config.
        """
        audio = self._decode_audio(data)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])

        # Feed to VAD
        vad_events = self.vad.feed(audio)

        # Process VAD events
        for event in vad_events:
            if event.type == VADEventType.SPEECH_START:
                self._speech_start_sample = event.timestamp_samples
            elif event.type == VADEventType.SPEECH_END:
                if self._speech_start_sample is not None:
                    self._pending_segments.append(
                        (self._speech_start_sample, event.timestamp_samples)
                    )
                    self._speech_start_sample = None

        # Transcribe completed segments
        return self._transcribe_pending()

    def feed_audio_float(self, audio: np.ndarray) -> List[TranscriptEvent]:
        """
        Feed float32 numpy audio directly (for file-based streaming).
        """
        self.audio_buffer = np.concatenate([self.audio_buffer, audio.astype(np.float32)])

        vad_events = self.vad.feed(audio)

        for event in vad_events:
            if event.type == VADEventType.SPEECH_START:
                self._speech_start_sample = event.timestamp_samples
            elif event.type == VADEventType.SPEECH_END:
                if self._speech_start_sample is not None:
                    self._pending_segments.append(
                        (self._speech_start_sample, event.timestamp_samples)
                    )
                    self._speech_start_sample = None

        return self._transcribe_pending()

    def flush(self) -> List[TranscriptEvent]:
        """
        End of stream. Transcribe any remaining audio.
        """
        events: List[TranscriptEvent] = []

        # If VAD has an open speech segment, close it
        if self._speech_start_sample is not None:
            end_sample = self._buffer_start_sample + len(self.audio_buffer)
            self._pending_segments.append((self._speech_start_sample, end_sample))
            self._speech_start_sample = None

        # Transcribe any pending segments
        events.extend(self._transcribe_pending())

        # If there's still unbounded audio (no VAD segments were created),
        # transcribe the remaining buffer if it's long enough
        remaining = len(self.audio_buffer)
        if remaining >= int(0.3 * self.sample_rate) and not events:
            t0 = time.perf_counter()
            text = self._transcribe_segment(self.audio_buffer)
            latency_ms = (time.perf_counter() - t0) * 1000
            if text.strip():
                self._segment_id += 1
                events.append(TranscriptEvent(
                    type="transcript.final",
                    text=text,
                    segment_id=self._segment_id,
                    timestamp_ms=self._elapsed_ms(),
                    audio_start_ms=self._buffer_start_sample / self.sample_rate * 1000,
                    audio_end_ms=(self._buffer_start_sample + remaining) / self.sample_rate * 1000,
                    latency_ms=latency_ms,
                ))

        self._reset_buffer()
        self.vad.reset()
        return events

    def _transcribe_pending(self) -> List[TranscriptEvent]:
        """Transcribe all pending (completed) segments."""
        events: List[TranscriptEvent] = []

        while self._pending_segments:
            start_sample, end_sample = self._pending_segments.pop(0)

            # VAD timestamps are absolute session offsets, while audio_buffer is a sliding
            # window that gets trimmed after each transcription.
            buffer_end_sample = self._buffer_start_sample + len(self.audio_buffer)
            start_sample = max(self._buffer_start_sample, min(start_sample, buffer_end_sample))
            end_sample = max(start_sample, min(end_sample, buffer_end_sample))

            rel_start = start_sample - self._buffer_start_sample
            rel_end = end_sample - self._buffer_start_sample

            segment_audio = self.audio_buffer[rel_start:rel_end]

            # Skip very short segments
            min_samples = int(self.config.vad.min_speech_duration_ms / 1000 * self.sample_rate)
            if len(segment_audio) < min_samples:
                continue

            t0 = time.perf_counter()
            text = self._transcribe_segment(segment_audio)
            latency_ms = (time.perf_counter() - t0) * 1000

            if text.strip():
                self._segment_id += 1
                events.append(TranscriptEvent(
                    type="transcript.final",
                    text=text,
                    segment_id=self._segment_id,
                    timestamp_ms=self._elapsed_ms(),
                    audio_start_ms=start_sample / self.sample_rate * 1000,
                    audio_end_ms=end_sample / self.sample_rate * 1000,
                    latency_ms=latency_ms,
                ))

            # Trim buffer up to end of this segment to free memory
            self.audio_buffer = self.audio_buffer[rel_end:]
            self._buffer_start_sample = end_sample

        return events

    def _transcribe_segment(self, segment: np.ndarray) -> str:
        """Run ASR inference on an audio segment."""
        tmp = f"/tmp/_stt_seg_{uuid.uuid4().hex}.wav"
        try:
            sf.write(tmp, segment.astype(np.float32), self.sample_rate)
            out = self.asr_model.transcribe([tmp], batch_size=1, verbose=False)
            return _extract_text(out).strip()
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _decode_audio(self, data: bytes) -> np.ndarray:
        """Decode raw bytes to float32 audio according to session config."""
        if self.config.encoding == AudioEncoding.PCM_S16LE:
            pcm = np.frombuffer(data, dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0
        elif self.config.encoding == AudioEncoding.PCM_F32LE:
            return np.frombuffer(data, dtype=np.float32).copy()
        else:
            # Default: assume int16
            pcm = np.frombuffer(data, dtype=np.int16)
            return pcm.astype(np.float32) / 32768.0

    def _elapsed_ms(self) -> float:
        return (time.time() - self._session_start) * 1000

    def _reset_buffer(self) -> None:
        self.audio_buffer = np.array([], dtype=np.float32)
        self._buffer_start_sample = 0
        self._speech_start_sample = None
        self._pending_segments = []
