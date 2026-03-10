#!/usr/bin/env python3
"""
Run streaming STT WebSocket server.

    python run_streaming_server.py

Then connect via Postman: ws://localhost:8000/ws/transcribe

Protocol:
- Send JSON: {"audio": "<base64 PCM int16>"}  (16kHz mono)
- Send JSON: {"end": true} when done
- Receive: {"text": "...", "is_final": false/true}
"""

import sys


def main():
    try:
        import uvicorn
    except ImportError:
        print("Install: pip install fastapi uvicorn websockets")
        sys.exit(1)

    from stt_streaming_server import app

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
