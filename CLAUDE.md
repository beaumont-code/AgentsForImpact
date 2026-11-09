# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Third Eye** — a voice-only macOS desktop app for visually impaired users. Uses the webcam as a surrogate eye: users speak to an AI agent that captures/interprets the visual scene, describes surroundings, reads text/signs, and gives walking directions — all through voice.

Built for the **NVIDIA AI Agent Hackathon**. Team of 2: Person A handles Mac hardware (camera, mic, speakers), Person B handles NVIDIA AI model integrations.

## Architecture

State machine: `LISTENING -> THINKING -> SPEAKING -> LISTENING`

- **Orchestrator** (`orchestrator.py`) — central coordinator, owns conversation history, dispatches tool calls, manages continuous mode background thread, mutes mic during TTS
- **Nemotron Agent** (`agent.py`) — text-only reasoning LLM via OpenAI SDK pointed at NVIDIA NIM. Orchestrates via tool calling (capture_and_describe, get_directions, toggle_continuous_mode, read_text). Agentic loop: call LLM -> detect tool_calls -> dispatch -> feed results back -> repeat
- **Vision** (`vision.py`) — webcam capture via OpenCV + scene analysis via Llama 3.2 90B Vision (separate API call, results fed back to Nemotron as text)
- **Speech** (`speech.py`) — Riva ASR (streaming gRPC) + Riva TTS (batch gRPC), with fallbacks to SpeechRecognition + pyttsx3
- **Navigation** (`navigation.py`) — Apple MapKit via PyObjC (CLGeocoder + MKDirections), with Google Maps fallback

Key design decisions:
- Synchronous + threading (not asyncio) for simpler PyObjC/gRPC compatibility
- Full conversation history passed each Nemotron call (1M context window)
- Nemotron is text-only; all vision goes through Llama Vision as a separate call

## NVIDIA NIM APIs

All models use the same `NVIDIA_API_KEY` (env var, loaded from `.env`):

| Model | Endpoint | SDK |
|---|---|---|
| Nemotron Super 120B | `https://integrate.api.nvidia.com/v1/chat/completions` | OpenAI SDK |
| Llama 3.2 90B Vision | `https://integrate.api.nvidia.com/v1/chat/completions` | OpenAI SDK |
| Riva ASR (Parakeet CTC 1.1B) | `grpc.nvcf.nvidia.com:443` | nvidia-riva-client (gRPC) |
| Riva TTS (FastPitch HiFi-GAN) | `grpc.nvcf.nvidia.com:443` | nvidia-riva-client (gRPC) |

## Commands

```bash
# System prerequisites (macOS)
brew install portaudio
xcode-select --install   # for PyObjC compilation

# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run
python main.py                # Normal mode (Riva speech)
python main.py --no-riva      # Fallback ASR/TTS (no Riva needed)

# Tests
pytest tests/
pytest tests/test_vision.py   # Single test file
```

## Environment

Requires `.env` file with `NVIDIA_API_KEY=nvapi-xxx` (get from build.nvidia.com). See `.env.example` for template.

## Implementation Status

All core modules are implemented. `test_vision.py` is confirmed working end-to-end (webcam capture → Llama Vision API → text description).

## Known Issues & Fixes

### macOS camera warm-up delay (vision.py)
OpenCV on macOS via AVFoundation returns a blank/failed frame on the very first `camera.read()` call after opening. Fixed by adding a `time.sleep(0.5)` after `VideoCapture` initialization in `_get_camera()`.

### venv must be used — do not use system or conda Python
The project dependencies (`cv2`, `nvidia-riva-client`, etc.) are installed only in `venv/`. Always run with:
```bash
source venv/bin/activate
python main.py
# or directly:
venv/bin/python main.py
```

### Camera permissions on macOS
If OpenCV prints `not authorized to capture video`, go to **System Settings > Privacy & Security > Camera** and enable access for your terminal app, then restart the terminal.
