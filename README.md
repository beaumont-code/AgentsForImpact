# Third Eye

A voice-only macOS desktop app for visually impaired users. Uses the webcam as a surrogate eye — speak to an AI agent that captures and interprets the visual scene, describes surroundings, reads text/signs, and gives walking directions, all through voice.

Built for the **NVIDIA AI Agent Hackathon**.

---

## What It Does

- **Describe surroundings** — point the webcam and ask "what's around me?" to get a telegraphic spatial description with distance estimates (e.g. *"Person ahead, 4 feet. Wall left, 2 feet."*)
- **Read text and signs** — ask "read that sign" to extract and speak text visible in the scene with location and distance
- **Walking directions** — ask "how do I get to Starbucks?" to get step-by-step walking directions via Apple MapKit
- **Continuous monitoring** — enable continuous mode for hands-free background scanning that alerts you when the scene changes, with proximity detection for nearby obstacles

Everything is voice-in, voice-out. No screen interaction required.

---

## Demo

> *Demo video/GIF coming soon*

**Example voice interactions:**

```
User:   "What's in front of me?"
Agent:  "Clear hallway. Door right, 15 feet."

User:   "Read that sign."
Agent:  "'Exit' on sign ahead, 6 feet."

User:   "Take me to the library."
Agent:  "Head north on Main Street. Turn right on Oak, 200 feet."

[Continuous mode — automatic alerts]
Agent:  "Person ahead, 3 feet."
```

---

## Architecture

```
State Machine:  LISTENING -> THINKING -> SPEAKING -> LISTENING
```

```
+======================================================================+
|                        THIRD EYE (Python, macOS)                     |
+======================================================================+
|                                                                      |
|  +------------------+        +-----------------------------------+   |
|  |   MICROPHONE     |        |         SPEAKERS                  |   |
|  |   (sounddevice)  |        |         (sounddevice)             |   |
|  +--------+---------+        +----------------^------------------+   |
|           |                                   |                      |
|           v                                   |                      |
|  +--------+---------+        +----------------+------------------+   |
|  |   RIVA ASR       |        |         RIVA TTS                  |   |
|  |   Speech-to-Text |        |         Text-to-Speech            |   |
|  |   gRPC streaming |        |         gRPC batch                |   |
|  +--------+---------+        +----------------^------------------+   |
|           |                                   |                      |
|           v                                   |                      |
|  +--------+-------------------------------------------+----------+   |
|  |                ORCHESTRATOR (orchestrator.py)                  |   |
|  |                                                                |   |
|  |  State Machine: LISTENING -> THINKING -> SPEAKING -> LISTENING |   |
|  |                                                                |   |
|  |  - Owns conversation_history: list[dict]                       |   |
|  |  - Dispatches Nemotron tool_calls to handlers                  |   |
|  |  - Manages continuous_mode background thread                   |   |
|  |  - Mutes mic during TTS playback                               |   |
|  +------+----------------+-------------------+-------------------+   |
|         |                |                   |                       |
|         v                v                   v                       |
|  +------+------+  +-----+-------+  +--------+----------+            |
|  |  NEMOTRON   |  |  WEBCAM +   |  |  NAVIGATION       |            |
|  |  AGENT      |  |  LLAMA      |  |  (Apple MapKit     |            |
|  |             |  |  VISION     |  |   via PyObjC)      |            |
|  | OpenAI SDK  |  |             |  |                    |            |
|  | Tool calling|  | cv2.Video   |  | CLGeocoder         |            |
|  |             |  | Capture +   |  | MKDirections       |            |
|  | Model:      |  | Llama 3.2   |  | Walking directions |            |
|  | nemotron-3  |  | 90B Vision  |  |                    |            |
|  | -super-120b |  | REST API    |  | Fallback: Google   |            |
|  | -a12b       |  |             |  | Maps Directions    |            |
|  +-------------+  +-------------+  +--------------------+            |
|                                                                      |
+======================================================================+
```

| Module | Role |
|---|---|
| `orchestrator.py` | Central coordinator — state machine, conversation history, tool dispatch, continuous mode, live camera overlay |
| `agent.py` | Nemotron 120B reasoning agent via NVIDIA NIM — agentic tool-calling loop |
| `vision.py` | Webcam capture (OpenCV) + Llama 3.2 90B Vision scene analysis |
| `speech.py` | Riva ASR (streaming gRPC) + Riva TTS (batch gRPC), with local fallbacks |
| `navigation.py` | Apple MapKit walking directions via PyObjC (CLGeocoder + MKDirections) |
| `config.py` | All constants, API endpoints, model IDs — loaded from `.env` |
| `prompts.py` | System prompt (Nemotron) and vision prompt templates (Llama Vision) |
| `audio_utils.py` | Low-level mic recording + speaker playback via sounddevice |

---

## NVIDIA NIM Models

All models run on NVIDIA's cloud via a single `NVIDIA_API_KEY`:

| Model | Endpoint | Purpose |
|---|---|---|
| **Nemotron Super 120B** | `integrate.api.nvidia.com/v1` (OpenAI SDK) | Text reasoning + tool calling — decides when to look, navigate, or read |
| **Llama 3.2 90B Vision** | `integrate.api.nvidia.com/v1` (OpenAI SDK) | Scene analysis — receives webcam frames, returns spatial descriptions |
| **Riva ASR** (Parakeet CTC 1.1B) | `grpc.nvcf.nvidia.com:443` (gRPC) | Speech-to-text — streaming microphone input |
| **Riva TTS** (FastPitch HiFi-GAN) | `grpc.nvcf.nvidia.com:443` (gRPC) | Text-to-speech — speaks responses to the user |

Key design: Nemotron is **text-only** and acts as the reasoning orchestrator. All vision goes through Llama Vision as a separate API call — results are fed back to Nemotron as text context.

---

## Prerequisites

- **macOS** (required — uses AVFoundation for camera, PyObjC for MapKit)
- **Python 3.11+**
- **NVIDIA API key** from [build.nvidia.com](https://build.nvidia.com) (free signup, 1,000 API credits included)

Install system dependencies:

```bash
brew install portaudio
xcode-select --install   # for PyObjC compilation
```

---

## Setup

**1. Clone and enter the repo:**

```bash
git clone <repo-url>
cd AgentsForImpact
```

**2. Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate
```

> **Note:** If you have conda installed, `python` may still point to conda even with the venv active. Always use `venv/bin/python` to run scripts directly, or deactivate conda first with `conda deactivate`.

**3. Install dependencies:**

```bash
venv/bin/pip install -r requirements.txt
```

**4. Create your `.env` file:**

```bash
cp .env.example .env
```

Then edit `.env` and add your key:

```
NVIDIA_API_KEY=nvapi-xxx
```

**5. Grant camera and microphone permissions:**

Go to **System Settings > Privacy & Security** and enable Camera and Microphone access for your terminal app. Restart the terminal after granting.

---

## Usage

> Always use `venv/bin/python` instead of `python` to ensure the venv is used.

**Normal mode** — full voice interaction with Riva ASR/TTS:

```bash
venv/bin/python main.py
```

**Fallback mode** — uses local SpeechRecognition + pyttsx3 instead of Riva (no Riva access needed):

```bash
venv/bin/python main.py --no-riva
```

**Vision-only mode** — no mic input, auto-starts continuous mode with camera feed and TTS alerts:

```bash
venv/bin/python main.py --vision-only --no-riva
```

In normal mode, speak naturally. Say **"exit"** or **"quit"** to stop.

---

## How It Works

1. **User speaks** — Riva ASR (or SpeechRecognition fallback) converts speech to text
2. **Nemotron reasons** — the 120B reasoning model receives the text + full conversation history and decides what to do via tool calling:
   - `capture_and_describe(focus?)` — capture a webcam frame and describe the scene
   - `read_text()` — extract visible text from the current view
   - `get_directions(destination, origin?)` — get walking directions via MapKit
   - `toggle_continuous_mode(enabled, interval?)` — start/stop background monitoring
3. **Tools execute** — vision frames go to Llama 3.2 90B Vision, navigation queries go to Apple MapKit
4. **Results fed back** — tool outputs return to Nemotron, which generates a final telegraphic response (3-5 words max)
5. **Response spoken** — Riva TTS (or pyttsx3 fallback) speaks the response aloud

**Continuous mode** runs as a background thread:
- Captures frames at a configurable interval (default 5s, 3s in vision-only mode)
- **Frame diff gate** — compares pixel MSE between frames, skips API calls if the scene barely changed
- **Text diff gate** — compares vision description word sets, only speaks if >50% of words differ
- **Proximity detection** — regex-matches distance patterns (e.g. "3 feet") and alerts on close obstacles

---

## Performance

Latency measured from 35 debug sessions in `--vision-only --no-riva` mode:

| Stage | Min | Typical | Max |
|---|---|---|---|
| Vision (Llama 3.2 90B) | 0.7s | 2–5s | 17.2s |
| Nemotron (120B)* | 0.8s | 1–3s | 15.2s |
| TTS (pyttsx3 fallback) | <0.1s | <0.1s | <0.1s |
| **Total (with Nemotron)** | 2.5s | 4–8s | 21.4s |
| **Total (vision-only)** | 0.7s | 2–5s | 17.2s |

*Nemotron was later bypassed in continuous mode to reduce latency.

**Notes:**
- Riva ASR/TTS latency could not be measured (service was unavailable during testing)
- Frame-diff gating (MSE) skips ~80% of unchanged frames, avoiding API calls
- Text-diff gating (word similarity) skips ~95% of redundant descriptions
- Vision API latency varies with scene complexity

---

## Project Structure

```
AgentsForImpact/
  main.py               # Entry point — CLI args, config overrides, runs orchestrator
  orchestrator.py        # State machine, conversation loop, continuous mode, camera overlay
  agent.py               # Nemotron agentic loop — LLM call -> tool dispatch -> repeat
  vision.py              # OpenCV webcam capture + Llama Vision API calls
  speech.py              # Riva ASR/TTS with SpeechRecognition/pyttsx3 fallbacks
  navigation.py          # Apple MapKit via PyObjC (CLGeocoder + MKDirections)
  config.py              # API keys, endpoints, model IDs, all constants
  prompts.py             # System prompt + vision prompt templates
  audio_utils.py         # Low-level sounddevice mic/speaker I/O
  test_vision.py         # Interactive vision pipeline test (webcam -> Llama Vision)
  camera_test.py         # Basic camera feed sanity check
  requirements.txt       # Python dependencies
  .env.example           # Template for NVIDIA_API_KEY
```

---

## Team

Built by a team of 2 for the NVIDIA AI Agent Hackathon:
- **Person A** — Mac hardware integration (camera, microphone, speakers, MapKit)
- **Person B** — NVIDIA AI model integrations (Nemotron, Llama Vision, Riva ASR/TTS)
