# Streaming STT API Contract

This service exposes a production-oriented websocket contract for low-latency speech-to-text using NVIDIA Parakeet plus server-side VAD.

It is intended to feed downstream LLM/agent systems with finalized speech segments over a stable JSON event stream.

Protocol version: `2026-03-10`

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `WebSocket` | `/ws/stt` | Primary realtime STT contract |
| `WebSocket` | `/ws/transcribe` | Legacy compatibility endpoint |
| `POST` | `/transcribe/file` | File upload, returns streaming NDJSON |
| `GET` | `/health` | Liveness and model status |
| `GET` | `/config/defaults` | Default websocket session config |
| `GET` | `/protocol` | Machine-readable websocket protocol metadata and schemas |

## Core Contract

The canonical flow is:

1. Connect to `/ws/stt`.
2. Send exactly one JSON `config` message first.
3. Stream binary audio frames or legacy base64 audio messages.
4. Receive ordered JSON events with `sequence` numbers.
5. Send `{ "type": "end" }` to flush the final segment.

Important constraints:

- The first websocket message must be a config object.
- Raw websocket audio currently supports only `16000 Hz`.
- Supported encodings are `pcm_s16le` and `pcm_f32le`.
- `interim_results` is currently unsupported.
- `language` override is currently unsupported.
- Use `transcript.final` as the LLM input boundary.

## Client Messages

### 1. Config

Required as the first message on every websocket session.

```json
{
  "type": "config",
  "config": {
    "sample_rate": 16000,
    "encoding": "pcm_s16le",
    "vad": {
      "enabled": true,
      "engine": "silero",
      "threshold": 0.4,
      "silence_duration_ms": 600,
      "min_speech_duration_ms": 250,
      "speech_pad_ms": 30
    },
    "interim_results": false,
    "language": null
  }
}
```

Notes:

- `sample_rate` must be `16000`.
- `interim_results` must be `false`.
- `language` must be `null`.

### 2. Binary Audio Frame

Send raw websocket binary frames after `session.created`.

- `pcm_s16le`: little-endian signed 16-bit mono PCM
- `pcm_f32le`: little-endian float32 mono PCM

Recommended frame sizing:

- `20-100 ms` frames for normal realtime use
- align to `512` samples when using Silero VAD-sensitive clients

### 3. End

Flushes buffered audio and terminates the session cleanly.

```json
{
  "type": "end"
}
```

### 4. Ping

Application-level keepalive / RTT correlation.

```json
{
  "type": "ping",
  "timestamp_ms": 1710065405123
}
```

### 5. Legacy Base64 Audio

Still supported for compatibility and tools like Postman, but binary frames are preferred in production.

```json
{
  "audio": "<base64 PCM bytes>"
}
```

## Server Events

Every server event on `/ws/stt` includes:

- `type`
- `protocol_version`
- `sequence`

`sequence` is strictly increasing per websocket session and should be used by downstream LLM consumers to preserve ordering.

### 1. session.created

Sent after a valid config is accepted.

```json
{
  "type": "session.created",
  "protocol_version": "2026-03-10",
  "sequence": 1,
  "session_id": "537e6e607bb1",
  "config": {
    "sample_rate": 16000,
    "encoding": "pcm_s16le",
    "vad": {
      "enabled": true,
      "engine": "silero",
      "threshold": 0.4,
      "silence_duration_ms": 600,
      "min_speech_duration_ms": 250,
      "speech_pad_ms": 30,
      "energy_threshold": 0.01
    },
    "language": null,
    "interim_results": false
  },
  "capabilities": {
    "binary_audio": true,
    "legacy_base64_audio": true,
    "ping_pong": true,
    "interim_results": false,
    "language_control": false,
    "supported_sample_rates_hz": [16000],
    "supported_audio_encodings": ["pcm_s16le", "pcm_f32le"]
  }
}
```

### 2. transcript.final

This is the primary event for LLM ingestion.

```json
{
  "type": "transcript.final",
  "protocol_version": "2026-03-10",
  "sequence": 4,
  "segment_id": 2,
  "text": "hello can you hear me",
  "timestamp_ms": 1841.7,
  "audio_start_ms": 250.0,
  "audio_end_ms": 1450.0,
  "latency_ms": 115.2,
  "confidence": 0.0
}
```

Semantics:

- `segment_id` is monotonically increasing within a session.
- `timestamp_ms` is relative to session start.
- `audio_start_ms` / `audio_end_ms` are relative to the full session audio timeline.
- `latency_ms` is ASR inference time for that completed segment.

### 3. pong

Sent in response to `ping`.

```json
{
  "type": "pong",
  "protocol_version": "2026-03-10",
  "sequence": 3,
  "timestamp_ms": 1710065405123
}
```

### 4. error

Structured protocol/runtime error.

```json
{
  "type": "error",
  "protocol_version": "2026-03-10",
  "sequence": 2,
  "session_id": "537e6e607bb1",
  "code": "config.invalid",
  "message": "Invalid session config.",
  "retryable": false,
  "details": {
    "validation_errors": []
  }
}
```

Important error codes:

- `config.required`
- `config.invalid`
- `config.already_set`
- `audio.unsupported_sample_rate`
- `message.invalid_json`
- `message.invalid`
- `message.unsupported_type`
- `audio.frame_too_large`
- `audio.buffer_overflow`
- `session.limit_reached`
- `internal_error`

### 5. session.ended

Sent after `end` processing is complete.

```json
{
  "type": "session.ended",
  "protocol_version": "2026-03-10",
  "sequence": 8,
  "session_id": "537e6e607bb1",
  "segments_transcribed": 4,
  "total_audio_s": 15.2,
  "session_duration_s": 16.0,
  "reason": "client_end"
}
```

## LLM Integration Guidance

Use the websocket stream like an append-only event log.

- Ignore everything except `transcript.final`, `error`, and `session.ended` in the LLM ingestion path.
- Preserve order using `sequence`.
- Treat `segment_id` as the unit of stable transcript progression.
- Do not feed raw partial hypotheses into an LLM context buffer.
- Persist `session_id`, `segment_id`, and `sequence` if you need replay/debug traceability.

Minimal downstream algorithm:

1. Wait for `session.created`.
2. Stream audio.
3. For each `transcript.final`, append `text` to the LLM conversation buffer.
4. On `error`, inspect `code` and reconnect if appropriate.
5. On `session.ended`, finalize the turn.

## File Upload Contract

`POST /transcribe/file` returns `application/x-ndjson`.

Each line is a JSON transcript object derived from the same segmentation pipeline as websocket mode.

Example:

```json
{"type":"transcript.final","text":"hello world","segment_id":1,"timestamp_ms":1000.0,"audio_start_ms":0.0,"audio_end_ms":1250.0,"latency_ms":140.0,"confidence":0.0,"is_final":false}
{"type":"transcript.final","text":"this is a test","segment_id":2,"timestamp_ms":2200.0,"audio_start_ms":1500.0,"audio_end_ms":2800.0,"latency_ms":160.0,"confidence":0.0,"is_final":true}
```

## Production Notes

These points matter before deployment:

- TLS termination is required for browser microphone clients (`wss://`).
- The `/ws/stt` contract is strict by design; malformed clients should reconnect rather than expect silent fallback behavior.
- Header rewriting for websocket `Host` / `Origin` is now an opt-in compatibility mode, not the default. Enable it only when a proxy genuinely requires it.
- The server enforces handshake timeout, max frame size, and max buffered audio duration to reduce abuse and unbounded memory growth.
- `/protocol` should be treated as the machine-readable source of truth for clients and orchestration layers.

## Current Non-Goals / Known Gaps

These are still outside the current implementation and should be addressed separately during deployment planning:

- Authentication / authorization
- Per-tenant rate limits
- Structured metrics / tracing export
- Durable transcript persistence
- Native partial transcript streaming
- Language-forcing controls
