#!/usr/bin/env python3
"""
Run streaming STT server.

    python run_streaming_server.py
    python run_streaming_server.py --fp16 --port 8000
    python run_streaming_server.py --model nvidia/parakeet-tdt-1.1b --device cuda

Endpoints:
    WebSocket: ws://localhost:8000/ws/stt        (new binary protocol)
    WebSocket: ws://localhost:8000/ws/transcribe  (legacy base64 protocol)
    HTTP:      POST http://localhost:8000/transcribe/file
    Info:      GET  http://localhost:8000/health
    Config:    GET  http://localhost:8000/config/defaults
"""

import argparse
import sys


def main():
    try:
        import uvicorn
    except ImportError:
        print("Install: pip install fastapi uvicorn websockets pydantic")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Streaming STT Server")
    parser.add_argument("--model", default="nvidia/parakeet-tdt-1.1b", help="ASR model name")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference (GPU only)")
    parser.add_argument("--max-sessions", type=int, default=10, help="Max concurrent WS sessions")
    args = parser.parse_args()

    from config import ServerConfig
    from stt_streaming_server import create_streaming_app

    server_config = ServerConfig(
        model_name=args.model,
        device=args.device,
        host=args.host,
        port=args.port,
        use_fp16=args.fp16,
        max_sessions=args.max_sessions,
    )

    app = create_streaming_app(server_config)
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port, 
        proxy_headers=True, 
        forwarded_allow_ips="*"
    )


if __name__ == "__main__":
    main()
