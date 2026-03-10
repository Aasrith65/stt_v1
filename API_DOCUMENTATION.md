# Real-Time STT Service API Documentation

Welcome to the Real-Time Streaming Speech-to-Text (STT) API. This service provides highly accurate, low-latency, real-time transcription powered by NVIDIA Parakeet and Silero VAD (Voice Activity Detection).

The API is specifically designed for integration with real-time generative AI voice agents, LLM orchestrators (like LangChain), and browser-based real-time communication.

---

## 🚀 Quick Overview

The service fundamentally operates using **WebSockets with a binary audio protocol**. 

1. **Connect** to the WebSocket endpoint.
2. **Send JSON config** to define audio format, language, and VAD preferences.
3. **Stream binary PCM audio** continuously. 
4. **Receive JSON events** asynchronously (such as `transcript.final`) whenever the VAD detects complete speech segments.

---

## 📡 Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `WebSocket` | `/ws/stt` | **[Core]** Real-time streaming transcription endpoint. |
| `POST` | `/transcribe/file` | Traditional REST endpoint for uploading and transcribing full audio files. |
| `GET` | `/health` | Server health, model loaded, and active session count. |
| `GET` | `/config/defaults`| The default session configuration schema. |

---

## 1. Real-Time Streaming (WebSocket)

**Endpoint:** `ws://<server_url>/ws/stt`  
*(Use `wss://` if the server is deployed behind HTTPS/TLS)*

### Connection Flow

1. **Establish WebSocket connection.**
2. **Configuration Phase:** The client **MUST** send a JSON configuration message immediately upon connection. If no configuration is sent, the server will close the connection.
3. **Streaming Phase:** Once configured, the client streams raw binary audio frames.
4. **Event Phase:** The server asynchronously emits JSON events indicating speech tracking and transcriptions.

### Step 1: Client Configuration Message

Upon connection, send the following JSON payload. All fields inside `config` are optional and will fall back to server defaults.

**Payload:**
```json
{
  "type": "config",
  "config": {
    "sample_rate": 16000,
    "encoding": "pcm_s16le",
    "vad": {
      "enabled": true,
      "engine": "silero",
      "silence_duration_ms": 800,
      "threshold": 0.5
    },
    "language": "en",
    "interim_results": true
  }
}
```

**Configuration Fields:**
- `sample_rate` (int): Audio sample rate in Hz. **Default: `16000`** (Currently only 16kHz is supported; others will be resampled, which introduces latency).
- `encoding` (string): Audio format. **Default: `pcm_s16le`** (16-bit signed integer PCM, little-endian).
- `vad` (object): Voice Activity Detection settings.
  - `enabled` (bool): Enable VAD chunking. **Default: `true`**.
  - `engine` (string): VAD algorithm (`"silero"`, `"energy"`, `"none"`). **Default: `"silero"`**.
  - `silence_duration_ms` (int): How much milliseconds of silence determines the "end" of a sentence/segment. **Default: `800`**. Lower values chunk faster; higher values keep thoughts connected.
  - `threshold` (float): Sensitivity of the VAD (0.0 to 1.0). **Default: `0.5`**.

### Step 2: Server Acknowledgment

The server will respond confirming the session creation and echoing the applied configuration:

```json
{
  "type": "session.created",
  "session_id": "c4d1b32a792f",
  "config": {
    "sample_rate": 16000,
    ...
  }
}
```

### Step 3: Streaming Audio

After receiving `session.created`, the client can begin sending audio.

- **Format:** Raw binary bytes (matching the configured `encoding` and `sample_rate`).
- **Framing:** You do NOT need to wrap the binary data in JSON. Send raw `ArrayBuffer` or `bytes` frames over the WebSocket.
- **Chunk Size:** Send chunks frequently for lowest latency (e.g., 100ms chunks = 3200 bytes per frame for 16kHz Int16).

### Step 4: Server Events (Transcripts)

The server will emit the following JSON events back to the client as speech is processed:

#### A. Transcript Partial
Fired (if `interim_results=true`) when speech is actively ongoing but hasn't finished yet. Useful for UI feedback.
```json
{
  "type": "transcript.partial",
  "text": "hello how are",
  "segment_id": 1,
  "timestamp_ms": 1500
}
```

#### B. Transcript Final
**[CRITICAL FOR LLMs]** Fired when the VAD engine detects the end of speech (e.g., after 800ms of silence). This text represents a complete, finalized segment ready for an LLM to consume.
```json
{
  "type": "transcript.final",
  "text": "hello how are you doing today?",
  "segment_id": 1,
  "timestamp_ms": 1850,
  "metrics": {
    "latency_ms": 230,
    "audio_duration_ms": 1900
  }
}
```

### Step 5: Orderly Disconnect

When the client is done streaming, it should send an End-Of-Stream JSON message to force the server to flush and transcribe any audio left in the buffers:

```json
{
  "type": "end"
}
```

The server flushes the audio, sends any final `transcript.final` events, and then closes the connection gracefully with:

```json
{
  "type": "session.ended",
  "segments_transcribed": 4,
  "total_audio_s": 15.2
}
```

---

## 2. File Transcription (REST API)

If you have pre-recorded audio files and don't need real-time WebSockets, you can upload them via HTTP.

**Endpoint:** `POST /transcribe/file`

**Content-Type:** `multipart/form-data`

**Request:**
```bash
curl -N -X POST https://<server_url>/transcribe/file \
  -F "file=@/path/to/audio/Record.wav"
```

**Response:**
The endpoint returns a streaming response (`application/x-ndjson`). It will stream JSON segment objects line-by-line as the file is chunked and transcribed.

```json
{"text": "hello world", "start_time": 0.0, "end_time": 1.25}
{"text": "this is a test", "start_time": 1.5, "end_time": 2.8}
```

---

## 3. Informational Endpoints

### Health Check
**Endpoint:** `GET /health`  
Returns the server status, active WebSocket sessions, and model information.

```json
{
  "status": "ok",
  "model": "nvidia/parakeet-tdt-1.1b",
  "device": "cuda",
  "active_sessions": 2,
  "fp16": true
}
```

### Config Schema
**Endpoint:** `GET /config/defaults`  
Returns the default schema for the WebSocket `SessionConfig`.

---

## 🛠 Integration Architecture Advice (For LLMs)

When integrating this service with an LLM (Large Language Model) to build a voice agent:

1. **Do not use `transcript.partial` for LLM input.** Interim translations jump around and will confuse the LLM. Use partials strictly for visual UI feedback only.
2. **Subscribe to `transcript.final`.** Wait for this event, then push the `text` string onto your LLM context buffer. 
3. **Control the interruption pace using VAD configs.** If your agent cuts the user off too quickly, increase the `vad.silence_duration_ms` to `1000` or `1200` to give the user more time to pause between sentences. If the agent feels too slow to respond, lower it to `500` or `600`.
4. **Match your mic capture format.** The absolute lowest latency is achieved when the client records native `pcm_s16le` at exactly `16000` Hz and sends it unmodified over the WebSocket, avoiding any CPU-intensive server-side resampling.
