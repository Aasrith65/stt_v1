"""
Streaming STT Server

- WebSocket: streaming audio via ws://.../ws/transcribe (base64 JSON)
- HTTP: POST /transcribe/file - upload audio file, receive streaming transcript chunks

Run: python run_streaming_server.py
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
import torch

# Lazy imports
nemo_asr = None


def _get_nemo():
    global nemo_asr
    if nemo_asr is None:
        import nemo.collections.asr as nemo_asr
    return nemo_asr


def _extract_text(output) -> str:
    if isinstance(output, tuple) and len(output) > 0:
        output = output[0]
    if not isinstance(output, (list, tuple)) or len(output) == 0:
        return str(output)
    first = output[0]
    if isinstance(first, list) and first:
        first = first[0]
    return first.text if hasattr(first, "text") else str(first)


# ── Silero VAD (streaming) ───────────────────────────────────────────────────

VAD_WINDOW = 512  # 32ms at 16kHz
SAMPLE_RATE = 16000


def load_silero_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils


def create_vad_iterator(model, utils, sample_rate: int = 16000):
    """Create streaming VAD iterator. Needs 512 samples per call at 16kHz."""
    VADIterator = utils[3]
    return VADIterator(model, sampling_rate=sample_rate)


# ── Streaming pipeline ────────────────────────────────────────────────────────

class StreamingSTTPipeline:
    """
    Per-connection pipeline: buffer → VAD → transcribe → send.
    """

    def __init__(self, asr_model, vad_model, vad_utils, device: str):
        self.asr_model = asr_model
        self.vad_model = vad_model
        self.device = device
        self.vad_iterator = create_vad_iterator(vad_model, vad_utils, SAMPLE_RATE)
        self.buffer = np.array([], dtype=np.float32)
        self._fed_samples = 0

    def add_audio(self, pcm_int16: np.ndarray) -> None:
        """Append PCM int16 to buffer. Converts to float32 [-1, 1]."""
        float32 = pcm_int16.astype(np.float32) / 32768.0
        self.buffer = np.concatenate([self.buffer, float32])

    def add_audio_float(self, audio_float: np.ndarray) -> None:
        """Append float32 [-1,1] audio to buffer."""
        self.buffer = np.concatenate([self.buffer, audio_float.astype(np.float32)])

    def process(self) -> list[str]:
        """
        Feed buffer to VAD. Return list of transcripts for completed segments.
        Trims buffer after each segment.
        """
        transcripts = []
        while len(self.buffer) - self._fed_samples >= VAD_WINDOW:
            chunk = self.buffer[self._fed_samples : self._fed_samples + VAD_WINDOW]
            self._fed_samples += VAD_WINDOW

            chunk_tensor = torch.from_numpy(chunk).float()
            speech_dict = self.vad_iterator(chunk_tensor, return_seconds=True)

            if speech_dict:
                start_s = speech_dict["start"]
                end_s = speech_dict["end"]
                start_idx = int(start_s * SAMPLE_RATE)
                end_idx = int(end_s * SAMPLE_RATE)
                end_idx = min(end_idx, len(self.buffer), self._fed_samples)
                start_idx = max(0, min(start_idx, end_idx - 1))

                if end_idx > start_idx:
                    segment = self.buffer[start_idx:end_idx]
                    text = self._transcribe_segment(segment)
                    if text.strip():
                        transcripts.append(text)

                self.vad_iterator.reset_states()
                self._trim_buffer(end_idx)
                self._fed_samples = 0

        return transcripts

    def flush(self) -> list[str]:
        """Flush remaining buffer. Treat rest as final segment if long enough."""
        transcripts = []
        self.process()
        if len(self.buffer) >= int(0.3 * SAMPLE_RATE):
            text = self._transcribe_segment(self.buffer)
            if text.strip():
                transcripts.append(text)
        self.buffer = np.array([], dtype=np.float32)
        self._fed_samples = 0
        self.vad_iterator.reset_states()
        return transcripts

    def _transcribe_segment(self, segment: np.ndarray) -> str:
        tmp = f"/tmp/_stt_seg_{uuid.uuid4().hex}.wav"
        try:
            sf.write(tmp, segment.astype(np.float32), SAMPLE_RATE)
            out = self.asr_model.transcribe([tmp], batch_size=1, verbose=False)
            return _extract_text(out).strip()
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _trim_buffer(self, keep_from: int) -> None:
        self.buffer = self.buffer[keep_from:].copy()


# ── FastAPI + WebSocket ─────────────────────────────────────────────────────

def _load_audio_file(file_path: str) -> np.ndarray:
    """Load audio file as 16kHz mono float32. Uses librosa for format support."""
    import librosa
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def create_streaming_app(
    model_name: str = "nvidia/parakeet-tdt-1.1b",
):
    try:
        from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
        from fastapi.responses import StreamingResponse
    except ImportError:
        raise ImportError("pip install fastapi uvicorn websockets")

    app = FastAPI(title="Streaming STT API")

    asr_model = None
    vad_model = None
    vad_utils = None

    @app.on_event("startup")
    async def startup():
        nonlocal asr_model, vad_model, vad_utils
        nemo = _get_nemo()
        print("Loading Parakeet...")
        asr_model = nemo.models.EncDecRNNTBPEModel.from_pretrained(model_name=model_name)
        asr_model = asr_model.to("cuda" if torch.cuda.is_available() else "cpu")
        asr_model.eval()
        asr_model.freeze()
        print("Loading Silero VAD...")
        vad_model, vad_utils = load_silero_vad()
        print("Ready.")

    @app.get("/")
    async def root():
        return {
            "status": "ok",
            "ws_url": "/ws/transcribe",
            "file_url": "POST /transcribe/file",
            "protocol": "Send JSON: {\"audio\": \"base64...\"} or {\"end\": true}",
        }

    @app.post("/transcribe/file")
    async def transcribe_file(request: Request):
        """
        Upload an audio file. Returns streaming transcript chunks (NDJSON).
        Each line: {"text": "...", "is_final": false/true}
        Body: multipart/form-data with key "file"
        """
        form = await request.form()
        file = form.get("file")
        if not file or not hasattr(file, "read"):
            raise HTTPException(status_code=400, detail="No file provided. Use form-data key 'file'.")
        suffix = ".wav"
        if getattr(file, "filename", None) and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            try:
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name

                device = "cuda" if torch.cuda.is_available() else "cpu"

                def generate():
                    try:
                        audio = _load_audio_file(tmp_path)
                        pipeline = StreamingSTTPipeline(asr_model, vad_model, vad_utils, device)
                        chunk_samples = int(1.0 * SAMPLE_RATE)
                        for i in range(0, len(audio), chunk_samples):
                            chunk = audio[i : i + chunk_samples]
                            pipeline.add_audio_float(chunk)
                            for t in pipeline.process():
                                yield json.dumps({"text": t, "is_final": False}) + "\n"
                        for t in pipeline.flush():
                            yield json.dumps({"text": t, "is_final": True}) + "\n"
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
                raise HTTPException(status_code=400, detail=str(e))

    @app.websocket("/ws/transcribe")
    async def ws_transcribe(websocket: WebSocket):
        await websocket.accept()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            await websocket.send_json({"status": "ready", "sample_rate": SAMPLE_RATE})
        except Exception:
            return

        pipeline = StreamingSTTPipeline(asr_model, vad_model, vad_utils, device)

        try:
            while True:
                msg = await websocket.receive()
                text_data = msg.get("text")
                bytes_data = msg.get("bytes")

                if text_data:
                    data = json.loads(text_data)
                    if data.get("end"):
                        for t in pipeline.flush():
                            await websocket.send_json({"text": t, "is_final": True})
                        break
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        pcm = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                        pipeline.add_audio(pcm)
                        for t in pipeline.process():
                            await websocket.send_json({"text": t, "is_final": False})

                elif bytes_data:
                    pcm = np.frombuffer(bytes_data, dtype=np.int16)
                    pipeline.add_audio(pcm)
                    for t in pipeline.process():
                        await websocket.send_json({"text": t, "is_final": False})

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({"error": str(e)})
            except Exception:
                pass

    return app


app = create_streaming_app()
