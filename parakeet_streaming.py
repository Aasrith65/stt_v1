# parakeet_streaming.py
"""Simple real‑time speech‑to‑text using NVIDIA Parakeet (TDT‑1.1B).

Run with:
    python parakeet_streaming.py

The script:
  * Loads the Parakeet model via NeMo.
  * Opens the default microphone (16 kHz mono) with PyAudio.
  * Sends audio in 1‑second chunks (adjust CHUNK_SEC).
  * Prints transcriptions as they become available.
  * Measures per‑chunk latency.

Prerequisites (install once in the notebook or terminal):
    pip install nemo_toolkit[asr] sounddevice pyaudio numpy torch
"""

import os, sys, time, numpy as np, torch
import pyaudio as pa
from nemo.collections.asr.models import EncDecRNNTBPEModel

# -------------------- 1️⃣ Config --------------------
MODEL_NAME = "nvidia/parakeet-tdt-1.1b"  # change to rnnt‑1.1b if you prefer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000  # Parakeet expects 16 kHz mono
CHUNK_SEC = 1.0      # length of each audio chunk (seconds)
OVERLAP_SEC = 0.2    # overlap to avoid word‑cutting

# -------------------- 2️⃣ Load model --------------------
print(f"Loading {MODEL_NAME} on {DEVICE} …")
asr = EncDecRNNTBPEModel.from_pretrained(MODEL_NAME).freeze()
asr = asr.to(DEVICE)
asr.eval()

# Adjust pre‑processor (same settings as the notebook)
cfg = asr._cfg
cfg.preprocessor.dither = 0.0
cfg.preprocessor.pad_to = 0
# Normalisation constants (copied from notebook – keep as‑is)
cfg.preprocessor.normalize = {
    "fixed_mean": [
        -14.95827016, -12.71798736, -11.76067913, -10.83311182,
        -10.6746914, -10.15163465, -10.05378331, -9.53918999,
        -9.41858904, -9.23382904, -9.46470918, -9.56037,
        -9.57434245, -9.47498732, -9.7635205, -10.08113074,
        -10.05454561, -9.81112681, -9.68673603, -9.83652977,
        -9.90046248, -9.85404766, -9.92560366, -9.95440354,
        -10.17162966, -9.90102482, -9.47471025, -9.54416855,
        -10.07109475, -9.98249912, -9.74359465, -9.55632283,
        -9.23399915, -9.36487649, -9.81791084, -9.56799225,
        -9.70630899, -9.85148006, -9.8594418, -10.01378735,
        -9.98505315, -9.62016094, -10.342285, -10.41070709,
        -10.10687659, -10.14536695, -10.30828702, -10.23542833,
        -10.88546868, -11.31723646, -11.46087382, -11.54877829,
        -11.62400934, -11.92190509, -12.14063815, -11.65130117,
        -11.58308531, -12.22214663, -12.42927197, -12.58039805,
        -13.10098969, -13.14345864, -13.31835645, -14.47345634
    ],
    "fixed_std": [
        3.81402054, 4.12647781, 4.05007065, 3.87790987,
        3.74721178, 3.68377423, 3.69344, 3.54001005,
        3.59530412, 3.63752368, 3.62826417, 3.56488469,
        3.53740577, 3.68313898, 3.67138151, 3.55707266,
        3.54919572, 3.55721289, 3.56723346, 3.46029304,
        3.44119672, 3.49030548, 3.39328435, 3.28244406,
        3.28001423, 3.26744937, 3.46692348, 3.35378948,
        2.96330901, 2.97663111, 3.04575148, 2.89717604,
        2.95659301, 2.90181116, 2.7111687, 2.93041291,
        2.86647897, 2.73473181, 2.71495654, 2.75543763,
        2.79174615, 2.96076456, 2.57376336, 2.68789782,
        2.90930817, 2.90412004, 2.76187531, 2.89905006,
        2.65896173, 2.81032176, 2.87769857, 2.84665271,
        2.80863137, 2.80707634, 2.83752184, 3.01914511,
        2.92046439, 2.78461139, 2.90034605, 2.94599508,
        2.99099718, 3.0167554, 3.04649716, 2.94116777
    ]
}
asr.preprocessor = asr.from_config_dict(cfg.preprocessor)

# -------------------- 3️⃣ Helper: Frame buffer --------------------
class FrameBuffer:
    """Keeps a sliding window with overlap.
    Returns a chunk ready for inference each time ``add`` is called.
    """
    def __init__(self, sr, chunk_sec, overlap_sec):
        self.sr = sr
        self.chunk_len = int(chunk_sec * sr)
        self.overlap_len = int(overlap_sec * sr)
        self.buffer = np.zeros(self.chunk_len + self.overlap_len, dtype=np.float32)
        self.filled = 0

    def add(self, audio):
        # audio is a 1‑D float32 array (already normalised to [-1,1])
        needed = self.chunk_len + self.overlap_len - self.filled
        if len(audio) < needed:
            # not enough yet – just store and wait
            self.buffer[self.filled:self.filled+len(audio)] = audio
            self.filled += len(audio)
            return None
        # fill the rest of the buffer
        self.buffer[self.filled:self.filled+needed] = audio[:needed]
        # prepare output chunk (last part of buffer)
        out = self.buffer.copy()
        # shift buffer left by chunk_len (keep overlap for next round)
        self.buffer[:self.overlap_len] = self.buffer[-self.overlap_len:]
        self.filled = self.overlap_len
        # return the part that will be fed to the model (without the overlap prefix)
        return out

# -------------------- 4️⃣ Audio callback --------------------
pa_instance = pa.PyAudio()
stream = pa_instance.open(
    format=pa.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=int(SAMPLE_RATE * CHUNK_SEC),
)

buf = FrameBuffer(SAMPLE_RATE, CHUNK_SEC, OVERLAP_SEC)
print("Listening… press Ctrl‑C to stop.")

try:
    while True:
        raw = stream.read(int(SAMPLE_RATE * CHUNK_SEC), exception_on_overflow=False)
        # Convert int16 PCM to float32 in [-1, 1]
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        chunk = buf.add(audio)
        if chunk is None:
            continue
        # -------------------- 5️⃣ Inference --------------------
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        start = time.time()
        # NeMo expects a batch; we give a single‑item batch
        logits, encoded_len, _ = asr.forward(
            input_signal=torch.tensor(chunk).unsqueeze(0).to(DEVICE),
            input_signal_length=torch.tensor([len(chunk)]).to(DEVICE),
        )
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        latency = (time.time() - start) * 1000  # ms
        # Greedy decode (same as notebook helper)
        pred = asr.decoding.ctc_decoder_predictions_tensor(logits)
        text = pred[0]
        print(f"[+{latency:.1f} ms] {text}")
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    stream.stop_stream()
    stream.close()
    pa_instance.terminate()
