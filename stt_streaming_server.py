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

import json
import os
import tempfile
import time
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from config import SessionConfig, ServerConfig, AudioEncoding
from pipeline import StreamingPipeline, TranscriptEvent, _extract_text, SAMPLE_RATE

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


# ── App Factory ──────────────────────────────────────────────────────────────

def create_streaming_app(server_config: Optional[ServerConfig] = None):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import StreamingResponse, JSONResponse
    except ImportError:
        raise ImportError("pip install fastapi uvicorn websockets")

    if server_config is None:
        server_config = ServerConfig()

    app = FastAPI(title="Streaming STT API", version="2.0.0")

    # Custom CORS middleware that doesn't interfere with WebSocket connections.
    # Starlette's CORSMiddleware can reject WebSocket upgrades with 403 in
    # certain versions, so we handle CORS manually for HTTP only.
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response as StarletteResponse

    @app.middleware("http")
    async def cors_middleware(request: Request, call_next):
        # Handle preflight
        if request.method == "OPTIONS":
            response = StarletteResponse(status_code=200)
        else:
            response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    asr_model = None
    device = server_config.resolve_device()
    active_sessions: dict = {}

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
        }

    @app.get("/config/defaults")
    async def config_defaults():
        return SessionConfig().model_dump()

    @app.get("/")
    async def root():
        return {
            "service": "Streaming STT API v2",
            "endpoints": {
                "websocket": "/ws/stt",
                "file_upload": "POST /transcribe/file",
                "health": "GET /health",
                "config": "GET /config/defaults",
            },
            "protocol": "Connect to /ws/stt, send JSON config, then binary PCM frames.",
        }

    # ── WebSocket: Real-time streaming STT ───────────────────────────────

    @app.websocket("/ws/stt")
    async def ws_stt(websocket: WebSocket):
        await websocket.accept()
        session_id = uuid.uuid4().hex[:12]

        # Check session limit
        if len(active_sessions) >= server_config.max_sessions:
            await websocket.send_json({
                "type": "error",
                "message": f"Max sessions ({server_config.max_sessions}) reached. Try again later.",
            })
            await websocket.close()
            return

        active_sessions[session_id] = time.time()

        try:
            # Step 1: Wait for config message (with timeout)
            config = SessionConfig()  # defaults
            try:
                first_msg = await websocket.receive()
                text_data = first_msg.get("text")
                if text_data:
                    msg = json.loads(text_data)
                    if msg.get("type") == "config" and "config" in msg:
                        config = SessionConfig(**msg["config"])
                    elif msg.get("type") == "config":
                        # Config at top level
                        config = SessionConfig(**{
                            k: v for k, v in msg.items() if k != "type"
                        })
                    else:
                        # Not a config message — it might be audio or something else.
                        # Use defaults and process this message below.
                        pass
            except Exception:
                pass  # Use defaults

            # Step 2: Create pipeline
            pipeline = StreamingPipeline(asr_model, config, device)

            # Step 3: Send session.created
            await websocket.send_json({
                "type": "session.created",
                "session_id": session_id,
                "config": config.model_dump(),
            })

            # Step 4: Stream audio
            segments_count = 0
            total_audio_samples = 0
            session_start = time.time()

            while True:
                msg = await websocket.receive()
                text_data = msg.get("text")
                bytes_data = msg.get("bytes")

                if bytes_data:
                    # Binary PCM frame — main path
                    total_audio_samples += len(bytes_data) // 2  # int16 = 2 bytes
                    events = pipeline.feed_audio(bytes_data)
                    for event in events:
                        segments_count += 1
                        await websocket.send_json(event.to_dict())

                elif text_data:
                    data = json.loads(text_data)
                    msg_type = data.get("type", "")

                    if msg_type == "end" or data.get("end"):
                        # End of stream
                        flush_events = pipeline.flush()
                        for event in flush_events:
                            segments_count += 1
                            await websocket.send_json(event.to_dict())

                        total_audio_s = total_audio_samples / SAMPLE_RATE
                        await websocket.send_json({
                            "type": "session.ended",
                            "session_id": session_id,
                            "segments_transcribed": segments_count,
                            "total_audio_s": round(total_audio_s, 2),
                            "session_duration_s": round(time.time() - session_start, 2),
                        })
                        break

                    elif "audio" in data:
                        # Legacy: base64-encoded audio (backward compatibility)
                        import base64
                        pcm_bytes = base64.b64decode(data["audio"])
                        total_audio_samples += len(pcm_bytes) // 2
                        events = pipeline.feed_audio(pcm_bytes)
                        for event in events:
                            segments_count += 1
                            await websocket.send_json(event.to_dict())

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
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

    return app


# ── Default app instance (for uvicorn stt_streaming_server:app) ──────────

app = create_streaming_app()
