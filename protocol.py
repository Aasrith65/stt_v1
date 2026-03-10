"""
WebSocket protocol models and metadata for the streaming STT service.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from config import SessionConfig


PROTOCOL_VERSION = "2026-03-10"


class ErrorCode(str, Enum):
    CONFIG_REQUIRED = "config.required"
    CONFIG_INVALID = "config.invalid"
    CONFIG_ALREADY_SET = "config.already_set"
    UNSUPPORTED_SAMPLE_RATE = "audio.unsupported_sample_rate"
    INVALID_JSON = "message.invalid_json"
    INVALID_MESSAGE = "message.invalid"
    UNSUPPORTED_MESSAGE_TYPE = "message.unsupported_type"
    FRAME_TOO_LARGE = "audio.frame_too_large"
    BUFFER_OVERFLOW = "audio.buffer_overflow"
    SESSION_LIMIT_REACHED = "session.limit_reached"
    INTERNAL_ERROR = "internal_error"


class ProtocolCapabilities(BaseModel):
    binary_audio: bool = True
    legacy_base64_audio: bool = True
    ping_pong: bool = True
    interim_results: bool = False
    language_control: bool = False
    supported_sample_rates_hz: List[int] = Field(default_factory=lambda: [16000])
    supported_audio_encodings: List[str] = Field(default_factory=lambda: ["pcm_s16le", "pcm_f32le"])


class ClientConfigMessage(BaseModel):
    type: Literal["config"]
    config: SessionConfig = Field(default_factory=SessionConfig)


class ClientEndMessage(BaseModel):
    type: Literal["end"]


class ClientPingMessage(BaseModel):
    type: Literal["ping"]
    timestamp_ms: Optional[float] = None


class LegacyAudioChunkMessage(BaseModel):
    audio: str


class SessionCreatedEvent(BaseModel):
    type: Literal["session.created"] = "session.created"
    session_id: str
    config: Dict[str, Any]
    protocol_version: str = PROTOCOL_VERSION
    sequence: int
    capabilities: ProtocolCapabilities


class SessionEndedEvent(BaseModel):
    type: Literal["session.ended"] = "session.ended"
    session_id: str
    segments_transcribed: int
    total_audio_s: float
    session_duration_s: float
    reason: str
    protocol_version: str = PROTOCOL_VERSION
    sequence: int


class PongEvent(BaseModel):
    type: Literal["pong"] = "pong"
    timestamp_ms: Optional[float] = None
    protocol_version: str = PROTOCOL_VERSION
    sequence: int


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: ErrorCode
    message: str
    retryable: bool = False
    session_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    protocol_version: str = PROTOCOL_VERSION
    sequence: int


def protocol_spec() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "capabilities": ProtocolCapabilities().model_dump(),
        "client_messages": {
            "config": ClientConfigMessage.model_json_schema(),
            "end": ClientEndMessage.model_json_schema(),
            "ping": ClientPingMessage.model_json_schema(),
            "legacy_audio_chunk": LegacyAudioChunkMessage.model_json_schema(),
        },
        "server_events": {
            "session.created": SessionCreatedEvent.model_json_schema(),
            "session.ended": SessionEndedEvent.model_json_schema(),
            "pong": PongEvent.model_json_schema(),
            "error": ErrorEvent.model_json_schema(),
        },
    }
