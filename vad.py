"""
Pluggable Voice Activity Detection (VAD) engines.

All engines implement the same interface:
    feed(chunk: np.ndarray) -> List[VADEvent]
    reset() -> None

Supported engines:
    - SileroVAD:  Neural VAD (best accuracy, ~2ms per 32ms window)
    - EnergyVAD:  Simple RMS energy-based (lowest latency, less accurate)
    - NoVAD:      Fixed-interval chunking (for pre-segmented or client-side VAD)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np
import torch

from config import VADConfig, VADEngine


# ── Events ───────────────────────────────────────────────────────────────────

class VADEventType(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class VADEvent:
    type: VADEventType
    timestamp_samples: int  # sample offset in the session's total audio
    timestamp_s: float      # seconds since session start


# ── Base class ───────────────────────────────────────────────────────────────

class BaseVAD(ABC):
    """Abstract base for all VAD engines."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.total_samples_fed: int = 0

    @abstractmethod
    def feed(self, chunk: np.ndarray) -> List[VADEvent]:
        """Feed float32 audio chunk, return VAD events."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new session."""
        ...

    def _make_event(self, event_type: VADEventType, sample_offset: int) -> VADEvent:
        return VADEvent(
            type=event_type,
            timestamp_samples=sample_offset,
            timestamp_s=sample_offset / self.sample_rate,
        )


# ── Silero VAD ───────────────────────────────────────────────────────────────

SILERO_WINDOW = 512  # 32ms at 16kHz — required by Silero


class SileroVAD(BaseVAD):
    """
    Silero VAD (streaming mode).
    Processes audio in 512-sample windows (32ms at 16kHz).
    Fires speech_start and speech_end events.
    """

    def __init__(self, config: VADConfig, sample_rate: int = 16000):
        super().__init__(sample_rate)
        self.config = config
        self._model = None
        self._vad_iterator = None
        self._in_speech = False
        self._speech_start_sample: int = 0
        self._leftover = np.array([], dtype=np.float32)
        self._load()

    def _load(self):
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self._model = model
        VADIterator = utils[3]
        self._vad_iterator = VADIterator(
            model,
            sampling_rate=self.sample_rate,
            threshold=self.config.threshold,
            min_silence_duration_ms=self.config.silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
        )

    def feed(self, chunk: np.ndarray) -> List[VADEvent]:
        events: List[VADEvent] = []
        audio = np.concatenate([self._leftover, chunk.astype(np.float32)])
        pos = 0

        while pos + SILERO_WINDOW <= len(audio):
            window = audio[pos: pos + SILERO_WINDOW]
            tensor = torch.from_numpy(window).float()
            result = self._vad_iterator(tensor, return_seconds=True)

            if result:
                if "start" in result:
                    start_s = result["start"]
                    start_sample = int(start_s * self.sample_rate)
                    self._in_speech = True
                    self._speech_start_sample = self.total_samples_fed + pos
                    events.append(self._make_event(
                        VADEventType.SPEECH_START,
                        self._speech_start_sample,
                    ))

                if "end" in result:
                    end_sample = self.total_samples_fed + pos + SILERO_WINDOW
                    self._in_speech = False
                    events.append(self._make_event(
                        VADEventType.SPEECH_END,
                        end_sample,
                    ))

            pos += SILERO_WINDOW

        self.total_samples_fed += pos
        self._leftover = audio[pos:]
        return events

    def reset(self) -> None:
        self.total_samples_fed = 0
        self._in_speech = False
        self._speech_start_sample = 0
        self._leftover = np.array([], dtype=np.float32)
        if self._vad_iterator:
            self._vad_iterator.reset_states()

    @property
    def in_speech(self) -> bool:
        return self._in_speech


# ── Energy VAD ───────────────────────────────────────────────────────────────

class EnergyVAD(BaseVAD):
    """
    Simple energy (RMS) based VAD.
    Fast, no model loading, but less accurate than Silero.
    """

    FRAME_SAMPLES = 1600  # 100ms at 16kHz

    def __init__(self, config: VADConfig, sample_rate: int = 16000):
        super().__init__(sample_rate)
        self.config = config
        self.threshold = config.energy_threshold
        self.silence_duration_s = config.silence_duration_ms / 1000.0
        self._in_speech = False
        self._silence_counter = 0.0
        self._speech_start_sample = 0
        self._leftover = np.array([], dtype=np.float32)

    def feed(self, chunk: np.ndarray) -> List[VADEvent]:
        events: List[VADEvent] = []
        audio = np.concatenate([self._leftover, chunk.astype(np.float32)])
        pos = 0
        frame_duration = self.FRAME_SAMPLES / self.sample_rate

        while pos + self.FRAME_SAMPLES <= len(audio):
            frame = audio[pos: pos + self.FRAME_SAMPLES]
            rms = float(np.sqrt(np.mean(frame ** 2)))
            abs_pos = self.total_samples_fed + pos

            if not self._in_speech:
                if rms > self.threshold:
                    self._in_speech = True
                    self._silence_counter = 0.0
                    self._speech_start_sample = abs_pos
                    events.append(self._make_event(VADEventType.SPEECH_START, abs_pos))
            else:
                if rms < self.threshold:
                    self._silence_counter += frame_duration
                else:
                    self._silence_counter = 0.0

                if self._silence_counter >= self.silence_duration_s:
                    self._in_speech = False
                    self._silence_counter = 0.0
                    events.append(self._make_event(VADEventType.SPEECH_END, abs_pos))

            pos += self.FRAME_SAMPLES

        self.total_samples_fed += pos
        self._leftover = audio[pos:]
        return events

    def reset(self) -> None:
        self.total_samples_fed = 0
        self._in_speech = False
        self._silence_counter = 0.0
        self._speech_start_sample = 0
        self._leftover = np.array([], dtype=np.float32)

    @property
    def in_speech(self) -> bool:
        return self._in_speech


# ── No VAD (fixed-interval) ─────────────────────────────────────────────────

class NoVAD(BaseVAD):
    """
    No VAD — emits speech_end every `interval_s` seconds.
    For use when the client handles VAD, or for batch-mode.
    """

    def __init__(self, config: VADConfig, sample_rate: int = 16000, interval_s: float = 2.0):
        super().__init__(sample_rate)
        self.interval_samples = int(interval_s * sample_rate)
        self._buffered = 0
        self._started = False

    def feed(self, chunk: np.ndarray) -> List[VADEvent]:
        events: List[VADEvent] = []

        if not self._started:
            self._started = True
            events.append(self._make_event(VADEventType.SPEECH_START, self.total_samples_fed))

        self._buffered += len(chunk)
        self.total_samples_fed += len(chunk)

        while self._buffered >= self.interval_samples:
            events.append(self._make_event(VADEventType.SPEECH_END, self.total_samples_fed))
            self._buffered -= self.interval_samples
            # Immediately start next segment
            events.append(self._make_event(VADEventType.SPEECH_START, self.total_samples_fed))

        return events

    def reset(self) -> None:
        self.total_samples_fed = 0
        self._buffered = 0
        self._started = False

    @property
    def in_speech(self) -> bool:
        return self._started


# ── Factory ──────────────────────────────────────────────────────────────────

def create_vad(config: VADConfig, sample_rate: int = 16000) -> BaseVAD:
    """Create a VAD engine from config."""
    if not config.enabled:
        return NoVAD(config, sample_rate)

    engine_map = {
        VADEngine.SILERO: SileroVAD,
        VADEngine.ENERGY: EnergyVAD,
        VADEngine.NONE: NoVAD,
    }
    cls = engine_map.get(config.engine, SileroVAD)
    return cls(config, sample_rate)
