"""
Streaming STT Server — Production Real-Time Service

Endpoints:
    WebSocket  /ws/stt           — Real-time streaming (binary PCM + JSON events)
    POST       /transcribe/file  — Upload audio file, receive streaming NDJSON
    GET        /health            — Health check
    GET        /config/defaults   — Default session configuration

Run: python run_streaming_server.py
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import tempfile
import time
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from pydantic import ValidationError

from config import SessionConfig, ServerConfig, AudioEncoding
from pipeline import StreamingPipeline, TranscriptEvent, _extract_text, SAMPLE_RATE
from protocol import (
    PROTOCOL_VERSION,
    ClientConfigMessage,
    ClientEndMessage,
    ClientPingMessage,
    ErrorCode,
    ErrorEvent,
    LegacyAudioChunkMessage,
    PongEvent,
    ProtocolCapabilities,
    SessionCreatedEvent,
    SessionEndedEvent,
    protocol_spec,
)

try:
    # FastAPI resolves postponed annotations from module globals, not factory locals.
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:  # pragma: no cover - handled by create_streaming_app()
    WebSocket = None  # type: ignore[assignment]
    WebSocketDisconnect = None  # type: ignore[assignment]

# Lazy NeMo import
nemo_asr = None


def _get_nemo():
    global nemo_asr
    if nemo_asr is None:
        import nemo.collections.asr as nemo_asr
    return nemo_asr


def _load_audio_file(file_path: str) -> np.ndarray:
    """Load audio file as 16kHz mono float32."""
    import librosa
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


class ProtocolError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        close_code: Optional[int] = 1008,
        retryable: bool = False,
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.close_code = close_code
        self.retryable = retryable
        self.details = details


def _parse_json_message(text_data: str) -> dict:
    try:
        data = json.loads(text_data)
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            ErrorCode.INVALID_JSON,
            "Text messages must be valid JSON.",
            details={"error": str(exc)},
        ) from exc

    if not isinstance(data, dict):
        raise ProtocolError(
            ErrorCode.INVALID_MESSAGE,
            "Text messages must decode to a JSON object.",
        )
    return data


def _parse_initial_config(message: dict) -> SessionConfig:
    if message.get("type") != "config":
        raise ProtocolError(
            ErrorCode.CONFIG_REQUIRED,
            "The first websocket message must be a config object: {\"type\":\"config\",\"config\":{...}}.",
        )

    try:
        if "config" in message:
            config = ClientConfigMessage.model_validate(message).config
        else:
            config = SessionConfig.model_validate({
                key: value for key, value in message.items() if key != "type"
            })
    except ValidationError as exc:
        raise ProtocolError(
            ErrorCode.CONFIG_INVALID,
            "Invalid session config.",
            details={"validation_errors": exc.errors()},
        ) from exc

    if config.sample_rate != SAMPLE_RATE:
        raise ProtocolError(
            ErrorCode.UNSUPPORTED_SAMPLE_RATE,
            f"Only {SAMPLE_RATE} Hz audio is supported on /ws/stt.",
            details={
                "requested_sample_rate": config.sample_rate,
                "supported_sample_rates_hz": [SAMPLE_RATE],
            },
        )

    if config.language is not None:
        raise ProtocolError(
            ErrorCode.CONFIG_INVALID,
            "language override is not supported by this websocket contract.",
            details={"field": "language"},
        )

    if config.interim_results:
        raise ProtocolError(
            ErrorCode.CONFIG_INVALID,
            "interim_results is not supported by this websocket contract.",
            details={"field": "interim_results"},
        )

    return config


def _frame_sample_count(data: bytes, encoding: AudioEncoding) -> int:
    if encoding == AudioEncoding.PCM_F32LE:
        return len(data) // 4
    return len(data) // 2


# ── App Factory ──────────────────────────────────────────────────────────────

def create_streaming_app(server_config: Optional[ServerConfig] = None):
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse, JSONResponse
    except ImportError:
        raise ImportError("pip install fastapi uvicorn websockets")

    if server_config is None:
        server_config = ServerConfig()

    app = FastAPI(title="Streaming STT API", version="2.0.0")

    # Enable simple CORS for browser HTTP clients
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"]
    )

    asr_model = None
    device = server_config.resolve_device()
    active_sessions: dict = {}
    capabilities = ProtocolCapabilities()
    max_buffer_samples = server_config.max_buffer_seconds * SAMPLE_RATE

    @app.on_event("startup")
    async def startup():
        nonlocal asr_model
        nemo = _get_nemo()
        print(f"Loading {server_config.model_name} on {device}...")
        asr_model = nemo.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=server_config.model_name
        )
        asr_model = asr_model.to(device)
        asr_model.eval()
        asr_model.freeze()

        if server_config.use_fp16 and device == "cuda":
            asr_model = asr_model.half()
            print("FP16 inference enabled.")

        print("Loading Silero VAD (warmup)...")
        # Warmup VAD so first connection is fast
        from vad import SileroVAD
        from config import VADConfig
        _warmup_vad = SileroVAD(VADConfig(), SAMPLE_RATE)
        del _warmup_vad

        print("Server ready.")

    # ── Health & Info ────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": server_config.model_name,
            "device": device,
            "active_sessions": len(active_sessions),
            "fp16": server_config.use_fp16,
            "protocol_version": PROTOCOL_VERSION,
        }

    @app.get("/config/defaults")
    async def config_defaults():
        return SessionConfig().model_dump()

    @app.get("/protocol")
    async def protocol():
        spec = protocol_spec()
        spec["defaults"] = SessionConfig().model_dump()
        spec["limits"] = {
            "config_timeout_s": server_config.config_timeout_s,
            "max_frame_bytes": server_config.max_frame_bytes,
            "max_buffer_seconds": server_config.max_buffer_seconds,
            "max_sessions": server_config.max_sessions,
        }
        return spec

    @app.get("/")
    async def root():
        return {
            "service": "Streaming STT API v2",
            "protocol_version": PROTOCOL_VERSION,
            "endpoints": {
                "websocket": "/ws/stt",
                "file_upload": "POST /transcribe/file",
                "health": "GET /health",
                "config": "GET /config/defaults",
                "protocol": "GET /protocol",
            },
            "protocol": "Connect to /ws/stt, send JSON config, then binary PCM frames.",
        }

    # ── WebSocket: Real-time streaming STT ───────────────────────────────

    @app.websocket("/ws/stt")
    async def ws_stt(websocket: WebSocket):
        await websocket.accept()
        session_id = uuid.uuid4().hex[:12]
        sequence = 0

        def next_sequence() -> int:
            nonlocal sequence
            sequence += 1
            return sequence

        async def send_event(payload: dict) -> None:
            await websocket.send_json({
                "protocol_version": PROTOCOL_VERSION,
                "sequence": next_sequence(),
                **payload,
            })

        async def send_error(
            code: ErrorCode,
            message: str,
            *,
            retryable: bool = False,
            details: Optional[dict] = None,
            close_code: Optional[int] = None,
        ) -> None:
            event = ErrorEvent(
                code=code,
                message=message,
                retryable=retryable,
                session_id=session_id,
                details=details,
                sequence=next_sequence(),
            )
            await websocket.send_json(event.model_dump())
            if close_code is not None:
                await websocket.close(code=close_code)

        # Check session limit
        if len(active_sessions) >= server_config.max_sessions:
            await send_error(
                ErrorCode.SESSION_LIMIT_REACHED,
                f"Max sessions ({server_config.max_sessions}) reached. Try again later.",
                retryable=True,
                close_code=1013,
            )
            return

        active_sessions[session_id] = time.time()

        try:
            try:
                first_msg = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=server_config.config_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                raise ProtocolError(
                    ErrorCode.CONFIG_REQUIRED,
                    f"Initial config message not received within {server_config.config_timeout_s:.1f}s.",
                ) from exc

            if first_msg.get("bytes") is not None:
                raise ProtocolError(
                    ErrorCode.CONFIG_REQUIRED,
                    "Binary audio cannot be sent before the initial config message.",
                )

            text_data = first_msg.get("text")
            if not text_data:
                raise ProtocolError(
                    ErrorCode.CONFIG_REQUIRED,
                    "The first websocket message must be a JSON config message.",
                )

            config = _parse_initial_config(_parse_json_message(text_data))
            pipeline = StreamingPipeline(asr_model, config, device)

            session_created = SessionCreatedEvent(
                session_id=session_id,
                config=config.model_dump(),
                sequence=next_sequence(),
                capabilities=capabilities,
            )
            await websocket.send_json(session_created.model_dump())

            segments_count = 0
            total_audio_samples = 0
            session_start = time.time()

            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break

                text_data = msg.get("text")
                bytes_data = msg.get("bytes")

                if bytes_data:
                    if len(bytes_data) > server_config.max_frame_bytes:
                        raise ProtocolError(
                            ErrorCode.FRAME_TOO_LARGE,
                            f"Audio frame exceeds max_frame_bytes={server_config.max_frame_bytes}.",
                            details={"frame_bytes": len(bytes_data)},
                        )

                    total_audio_samples += _frame_sample_count(bytes_data, config.encoding)
                    events = pipeline.feed_audio(bytes_data)
                    if len(pipeline.audio_buffer) > max_buffer_samples:
                        raise ProtocolError(
                            ErrorCode.BUFFER_OVERFLOW,
                            f"Buffered audio exceeded {server_config.max_buffer_seconds}s without a completed segment.",
                            details={"max_buffer_seconds": server_config.max_buffer_seconds},
                            close_code=1009,
                        )
                    for event in events:
                        segments_count += 1
                        await send_event(event.to_dict())

                elif text_data:
                    data = _parse_json_message(text_data)
                    msg_type = data.get("type", "")

                    if msg_type == "end" or data.get("end"):
                        ClientEndMessage.model_validate({"type": "end"})
                        flush_events = pipeline.flush()
                        for event in flush_events:
                            segments_count += 1
                            await send_event(event.to_dict())

                        total_audio_s = total_audio_samples / config.sample_rate
                        ended = SessionEndedEvent(
                            session_id=session_id,
                            segments_transcribed=segments_count,
                            total_audio_s=round(total_audio_s, 2),
                            session_duration_s=round(time.time() - session_start, 2),
                            reason="client_end",
                            sequence=next_sequence(),
                        )
                        await websocket.send_json(ended.model_dump())
                        break

                    elif msg_type == "ping":
                        ping = ClientPingMessage.model_validate(data)
                        pong = PongEvent(
                            timestamp_ms=ping.timestamp_ms,
                            sequence=next_sequence(),
                        )
                        await websocket.send_json(pong.model_dump())

                    elif msg_type == "config":
                        raise ProtocolError(
                            ErrorCode.CONFIG_ALREADY_SET,
                            "Config may only be sent once, before audio streaming begins.",
                        )

                    elif "audio" in data:
                        try:
                            legacy_audio = LegacyAudioChunkMessage.model_validate(data)
                            pcm_bytes = base64.b64decode(legacy_audio.audio, validate=True)
                        except (ValidationError, binascii.Error) as exc:
                            raise ProtocolError(
                                ErrorCode.INVALID_MESSAGE,
                                "Legacy base64 audio message is invalid.",
                                details={"error": str(exc)},
                            ) from exc

                        if len(pcm_bytes) > server_config.max_frame_bytes:
                            raise ProtocolError(
                                ErrorCode.FRAME_TOO_LARGE,
                                f"Audio frame exceeds max_frame_bytes={server_config.max_frame_bytes}.",
                                details={"frame_bytes": len(pcm_bytes)},
                            )

                        total_audio_samples += _frame_sample_count(pcm_bytes, config.encoding)
                        events = pipeline.feed_audio(pcm_bytes)
                        if len(pipeline.audio_buffer) > max_buffer_samples:
                            raise ProtocolError(
                                ErrorCode.BUFFER_OVERFLOW,
                                f"Buffered audio exceeded {server_config.max_buffer_seconds}s without a completed segment.",
                                details={"max_buffer_seconds": server_config.max_buffer_seconds},
                                close_code=1009,
                            )
                        for event in events:
                            segments_count += 1
                            await send_event(event.to_dict())

                    else:
                        raise ProtocolError(
                            ErrorCode.UNSUPPORTED_MESSAGE_TYPE,
                            f"Unsupported message type: {msg_type or 'missing'}",
                            details={"supported_types": ["config", "ping", "end"]},
                        )

        except WebSocketDisconnect:
            pass
        except ProtocolError as e:
            try:
                await send_error(
                    e.code,
                    e.message,
                    retryable=e.retryable,
                    details=e.details,
                    close_code=e.close_code,
                )
            except Exception:
                pass
        except Exception as e:
            try:
                await send_error(
                    ErrorCode.INTERNAL_ERROR,
                    "Unhandled server error.",
                    retryable=False,
                    details={"error": str(e)},
                    close_code=1011,
                )
            except Exception:
                pass
        finally:
            active_sessions.pop(session_id, None)

    # ── Legacy WebSocket (backward compat) ───────────────────────────────

    @app.websocket("/ws/transcribe")
    async def ws_transcribe_legacy(websocket: WebSocket):
        """Legacy endpoint — redirects to new protocol internally."""
        await websocket.accept()
        session_id = uuid.uuid4().hex[:12]

        config = SessionConfig()
        pipeline = StreamingPipeline(asr_model, config, device)

        try:
            await websocket.send_json({"status": "ready", "sample_rate": SAMPLE_RATE})

            while True:
                msg = await websocket.receive()
                text_data = msg.get("text")
                bytes_data = msg.get("bytes")

                if text_data:
                    data = json.loads(text_data)
                    if data.get("end"):
                        for event in pipeline.flush():
                            await websocket.send_json({
                                "text": event.text, "is_final": True
                            })
                        break
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        import base64
                        pcm_bytes = base64.b64decode(audio_b64)
                        for event in pipeline.feed_audio(pcm_bytes):
                            await websocket.send_json({
                                "text": event.text, "is_final": False
                            })

                elif bytes_data:
                    for event in pipeline.feed_audio(bytes_data):
                        await websocket.send_json({
                            "text": event.text, "is_final": False
                        })

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"error": str(e)})
            except Exception:
                pass

    # ── HTTP: File upload with streaming response ─────────────────────

    async def _transcribe_file_handler(request):
        """Raw Starlette handler for file upload — streams NDJSON."""
        form = await request.form()
        file = form.get("file")
        if not file or not hasattr(file, "read"):
            return JSONResponse(
                {"detail": "No file provided. Use form-data key 'file'."},
                status_code=400,
            )

        suffix = ".wav"
        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]

        content = await file.read()

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            try:
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name

                def generate():
                    try:
                        audio = _load_audio_file(tmp_path)
                        config = SessionConfig()
                        pipeline = StreamingPipeline(asr_model, config, device)
                        chunk_samples = int(1.0 * SAMPLE_RATE)

                        pending = None
                        for i in range(0, len(audio), chunk_samples):
                            chunk = audio[i: i + chunk_samples]
                            events = pipeline.feed_audio_float(chunk)
                            for event in events:
                                if pending is not None:
                                    pending_dict = pending.to_dict()
                                    pending_dict["is_final"] = False
                                    yield json.dumps(pending_dict) + "\n"
                                pending = event

                        for event in pipeline.flush():
                            if pending is not None:
                                pending_dict = pending.to_dict()
                                pending_dict["is_final"] = False
                                yield json.dumps(pending_dict) + "\n"
                            pending = event

                        # Last transcript is final
                        if pending is not None:
                            pending_dict = pending.to_dict()
                            pending_dict["is_final"] = True
                            yield json.dumps(pending_dict) + "\n"
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                return StreamingResponse(
                    generate(),
                    media_type="application/x-ndjson",
                )
            except Exception as e:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
                return JSONResponse({"detail": str(e)}, status_code=400)

    from starlette.routing import Route
    app.router.routes.append(
        Route("/transcribe/file", _transcribe_file_handler, methods=["POST"])
    )

    if server_config.rewrite_websocket_headers:
        # Compatibility shim for certain proxies that inject problematic Host/Origin
        # values. Leave disabled in normal production deployments.
        class AbsoluteNoOriginMiddleware:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] == "websocket":
                    scope["headers"] = [
                        (k, v) for k, v in scope.get("headers", [])
                        if k.lower() not in (b"origin", b"host")
                    ]
                await self.app(scope, receive, send)

        return AbsoluteNoOriginMiddleware(app)

    return app

# ── Default app instance (for uvicorn stt_streaming_server:app) ──────────

app = create_streaming_app()
