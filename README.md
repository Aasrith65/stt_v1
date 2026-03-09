# 🎙️ Local Speech-to-Text with Whisper Base

Local speech recognition using [OpenAI Whisper Base](https://huggingface.co/openai/whisper-base) via HuggingFace Transformers. Runs entirely on your machine — no API keys, no internet required after initial model download.

## Quick Start

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Transcribe an audio file
python transcribe.py --file path/to/audio.wav

# 4. Or record from your microphone (5 seconds by default)
python transcribe.py --mic --duration 10
```

## Features

| Feature | Command |
|---|---|
| Transcribe a file | `python transcribe.py --file audio.wav` |
| Record & transcribe | `python transcribe.py --mic --duration 10` |
| Set language | `python transcribe.py --file audio.wav --language french` |
| Translate to English | `python transcribe.py --file french.mp3 --task translate` |
| Force CPU | `python transcribe.py --file audio.wav --device cpu` |

## Supported Audio Formats

WAV, MP3, FLAC, OGG, and any format supported by `librosa` / `soundfile`.

## Model Info

- **Model**: `openai/whisper-base` (74M parameters)
- **Multilingual**: Yes (99 languages)
- **Tasks**: Speech recognition + translation to English
- **First run** downloads ~290 MB of model weights (cached for future use)
