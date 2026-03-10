"""
Configuration models for the STT streaming service.

Client-configurable session settings (VAD, encoding, language)
and server-level config (model, device, concurrency limits).
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class VADEngine(str, Enum):
    SILERO = "silero"
    ENERGY = "energy"
    NONE = "none"


class AudioEncoding(str, Enum):
    PCM_S16LE = "pcm_s16le"     # 16-bit signed little-endian (default)
    PCM_F32LE = "pcm_f32le"     # 32-bit float little-endian


# ── VAD Config ───────────────────────────────────────────────────────────────

class VADConfig(BaseModel):
    enabled: bool = True
    engine: VADEngine = VADEngine.SILERO
    # Silero-specific
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="VAD speech probability threshold")
    silence_duration_ms: int = Field(800, ge=100, le=5000, description="Ms of silence to end a segment")
    min_speech_duration_ms: int = Field(250, ge=50, le=5000, description="Min speech length to transcribe")
    speech_pad_ms: int = Field(30, ge=0, le=500, description="Padding around speech segments")
    # Energy-based VAD
    energy_threshold: float = Field(0.01, ge=0.0, le=1.0, description="RMS energy threshold (energy VAD)")


# ── Session Config (sent by client on connect) ──────────────────────────────

class SessionConfig(BaseModel):
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    encoding: AudioEncoding = AudioEncoding.PCM_S16LE
    vad: VADConfig = Field(default_factory=VADConfig)
    language: Optional[str] = Field(None, description="Language code (e.g. 'en'). None = auto-detect")
    interim_results: bool = Field(True, description="Send partial transcripts during speech")
    model_config = {"extra": "ignore"}


# ── Server Config (env vars / startup) ───────────────────────────────────────

class ServerConfig(BaseModel):
    model_name: str = Field(
        default_factory=lambda: os.environ.get("STT_MODEL", "nvidia/parakeet-tdt-1.1b")
    )
    device: str = Field(
        default_factory=lambda: os.environ.get("STT_DEVICE", "auto"),
        description="'cuda', 'cpu', or 'auto'"
    )
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    max_sessions: int = Field(
        default=10,
        description="Max concurrent WebSocket sessions"
    )
    use_fp16: bool = Field(
        default=False,
        description="Use FP16 inference (faster on GPU, may reduce accuracy)"
    )

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# ── Default configs ──────────────────────────────────────────────────────────

DEFAULT_SESSION_CONFIG = SessionConfig()
DEFAULT_SERVER_CONFIG = ServerConfig()
