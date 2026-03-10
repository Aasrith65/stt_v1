"""
Production STT Inference — Optimized for numerous concurrent calls

Features:
- Dynamic batching: Collects requests, processes in batches (higher throughput)
- Thread-safe: Single worker, queue-based
- Optional: torch.compile for faster inference (PyTorch 2.0+)

Usage:
    # Simple sync
    engine = STTInferenceEngine(model_name="nvidia/parakeet-tdt-1.1b")
    text = engine.transcribe("audio.wav")

    # Concurrent (batched)
    with engine:
        futures = [engine.submit(p) for p in paths]
        results = [f.result() for f in futures]

    # HTTP server
    uvicorn stt_inference:app --host 0.0.0.0 --port 8000
    curl -X POST -F "file=@audio.wav" http://localhost:8000/transcribe

Optimization options:
- batch_size: 8–16 for GPU (higher = more throughput, more memory)
- max_wait_ms: 50–100 (lower = faster latency, less batching)
- use_torch_compile: True (PyTorch 2.0+; faster after warmup)
"""

from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import soundfile as sf
import torch

# Lazy import NeMo (heavy)
nemo_asr = None


def _get_nemo():
    global nemo_asr
    if nemo_asr is None:
        import nemo.collections.asr as nemo_asr
    return nemo_asr


def _extract_text(output: Any) -> str:
    if isinstance(output, tuple) and len(output) > 0:
        output = output[0]
    if not isinstance(output, (list, tuple)) or len(output) == 0:
        return str(output)
    first = output[0]
    if isinstance(first, list) and first:
        first = first[0]
    return first.text if hasattr(first, "text") else str(first)


@dataclass
class _Request:
    req_id: str
    path: str
    result_queue: queue.Queue
    created_at: float = field(default_factory=time.time)


class STTInferenceEngine:
    """
    Production inference engine with dynamic batching.
    Thread-safe. Use transcribe() for simple sync calls, or submit()/get_result() for batching.
    """

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-1.1b",
        batch_size: int = 8,
        max_wait_ms: float = 50.0,
        use_torch_compile: bool = False,
        sample_rate: int = 16000,
    ):
        """
        Args:
            model_name: NeMo ASR model
            batch_size: Max items per batch (higher = more throughput, more GPU memory)
            max_wait_ms: Max ms to wait before processing partial batch
            use_torch_compile: PyTorch 2.0+ compile (faster after warmup)
            sample_rate: Expected audio sample rate
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.use_torch_compile = use_torch_compile
        self.sample_rate = sample_rate

        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._request_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def load_model(self) -> None:
        """Load model (call once before transcribe)."""
        nemo_asr = _get_nemo()
        print(f"Loading {self.model_name} on {self._device}...")
        t0 = time.time()
        self._model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_name=self.model_name
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._model.freeze()

        if self.use_torch_compile and hasattr(torch, "compile"):
            self._model.transcribe = torch.compile(
                self._model.transcribe, mode="reduce-overhead"
            )

        # Warmup
        if self._device == "cuda":
            warmup = np.zeros(self.sample_rate, dtype=np.float32)
            tmp = "/tmp/_stt_warmup.wav"
            sf.write(tmp, warmup, self.sample_rate)
            _ = self._model.transcribe([tmp], batch_size=1, verbose=False)
            torch.cuda.synchronize()
            try:
                os.remove(tmp)
            except OSError:
                pass

        print(f"Model loaded in {time.time()-t0:.1f}s")

    def _worker_loop(self) -> None:
        """Process batches from queue."""
        batch: List[_Request] = []
        last_flush = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                # Get next request (block up to 10ms)
                req = self._request_queue.get(timeout=0.01)
                batch.append(req)

                # Flush if: batch full OR max_wait elapsed
                now = time.perf_counter()
                wait_elapsed_ms = (now - last_flush) * 1000
                if len(batch) >= self.batch_size or wait_elapsed_ms >= self.max_wait_ms:
                    self._process_batch(batch)
                    batch = []
                    last_flush = now
            except queue.Empty:
                # Flush partial batch if max_wait elapsed
                if batch:
                    now = time.perf_counter()
                    if (now - last_flush) * 1000 >= self.max_wait_ms:
                        self._process_batch(batch)
                        batch = []
                        last_flush = now
            except Exception as e:
                for r in batch:
                    try:
                        r.result_queue.put(("", str(e)))
                    except Exception:
                        pass
                batch = []

    def _process_batch(self, requests: List[_Request]) -> None:
        if not requests or self._model is None:
            return
        paths = [r.path for r in requests]
        try:
            if self._device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = self._model.transcribe(paths, batch_size=len(paths), verbose=False)
            if self._device == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # NeMo returns list of hypotheses/texts in same order as paths
            if isinstance(output, tuple):
                output = output[0]
            texts = []
            for i, item in enumerate(output):
                if hasattr(item, "text"):
                    texts.append(item.text)
                elif isinstance(item, (list, tuple)) and item:
                    first = item[0]
                    texts.append(first.text if hasattr(first, "text") else str(first))
                else:
                    texts.append(str(item))

            for req, text in zip(requests, texts):
                req.result_queue.put((text, None))
        except Exception as e:
            for req in requests:
                try:
                    req.result_queue.put(("", str(e)))
                except Exception:
                    pass

    def submit(self, path: str, timeout: float = 60.0) -> "FutureResult":
        """
        Submit a request. Returns a Future-like object. Call .result() to get text.
        Thread-safe. Batched with other concurrent submits.
        """
        if self._model is None:
            self.load_model()
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

        result_q: queue.Queue = queue.Queue()
        req = _Request(req_id=str(uuid.uuid4()), path=path, result_queue=result_q)
        self._request_queue.put(req)
        return FutureResult(result_q, timeout)

    def transcribe(self, path: str, timeout: float = 60.0) -> str:
        """
        Synchronous transcribe. Blocks until done.
        For single calls. For concurrent calls, use submit() in a loop.
        """
        fut = self.submit(path, timeout)
        text, err = fut.result()
        if err:
            raise RuntimeError(err)
        return text

    def transcribe_audio(self, audio: np.ndarray, timeout: float = 60.0) -> str:
        """Transcribe from numpy array (writes temp file)."""
        tmp = f"/tmp/_stt_{uuid.uuid4().hex}.wav"
        try:
            sf.write(tmp, audio.astype(np.float32), self.sample_rate)
            return self.transcribe(tmp, timeout)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def __enter__(self) -> "STTInferenceEngine":
        self.load_model()
        return self

    def __exit__(self, *args) -> None:
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)


class FutureResult:
    """Simple future for submit() results."""

    def __init__(self, result_queue: queue.Queue, timeout: float):
        self._queue = result_queue
        self._timeout = timeout

    def result(self) -> tuple[str, Optional[str]]:
        """Returns (text, error). error is None on success."""
        try:
            text, err = self._queue.get(timeout=self._timeout)
            return (text or "", err)
        except queue.Empty:
            return ("", "Timeout waiting for result")


# ── FastAPI server (optional) ───────────────────────────────────────────────

def create_app(
    model_name: str = "nvidia/parakeet-tdt-1.1b",
    batch_size: int = 8,
    max_wait_ms: float = 50.0,
):
    """
    Create FastAPI app for HTTP STT serving.

    Run: uvicorn stt_inference:app --host 0.0.0.0 --port 8000
    Or:  app = create_app(); uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    try:
        from fastapi import FastAPI, File, UploadFile, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("pip install fastapi uvicorn python-multipart")

    app = FastAPI(title="STT Inference API")
    engine = STTInferenceEngine(
        model_name=model_name,
        batch_size=batch_size,
        max_wait_ms=max_wait_ms,
    )

    @app.on_event("startup")
    async def startup():
        engine.load_model()

    @app.post("/transcribe")
    async def transcribe_upload(file: UploadFile = File(...)):
        """Upload WAV file, get transcript."""
        import io
        import tempfile
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(contents)
            path = f.name
        try:
            text = engine.transcribe(path)
            return JSONResponse({"text": text})
        except Exception as e:
            raise HTTPException(500, str(e))
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# For: uvicorn stt_inference:app
app = create_app()
