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
    language: Optional[str] = Field(None, description="Reserved for future language override support. None = auto-detect")
    interim_results: bool = Field(False, description="Reserved for future partial transcript support. Currently unsupported.")
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
    config_timeout_s: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Seconds to wait for the initial websocket config message"
    )
    max_frame_bytes: int = Field(
        default=256_000,
        ge=512,
        description="Max websocket audio frame size in bytes"
    )
    max_buffer_seconds: int = Field(
        default=45,
        ge=1,
        le=600,
        description="Max untranscribed audio buffered per websocket session"
    )
    rewrite_websocket_headers: bool = Field(
        default_factory=lambda: os.environ.get("STT_REWRITE_WS_HEADERS", "").lower() in {"1", "true", "yes"},
        description="Compatibility hack for unusual proxies; strips Host/Origin before app routing. Keep disabled in production unless required."
    )

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


# ── Default configs ──────────────────────────────────────────────────────────

DEFAULT_SESSION_CONFIG = SessionConfig()
DEFAULT_SERVER_CONFIG = ServerConfig()
