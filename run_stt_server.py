#!/usr/bin/env python3
"""
Run STT inference server.

    python run_stt_server.py
    # Or: uvicorn stt_inference:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST -F "file=@audio.wav" http://localhost:8000/transcribe
"""

import sys


def main():
    try:
        import uvicorn
    except ImportError:
        print("Install: pip install fastapi uvicorn python-multipart")
        sys.exit(1)

    from stt_inference import create_app

    app = create_app(
        model_name="nvidia/parakeet-tdt-1.1b",
        batch_size=8,
        max_wait_ms=50.0,
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
