"""
All configuration constants for Third Eye.
Loads NVIDIA_API_KEY from .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not set. Create a .env file with NVIDIA_API_KEY=nvapi-xxx")

# --- Nemotron (Reasoning Agent) ---
NEMOTRON_BASE_URL = "https://integrate.api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_MAX_TOKENS = 2048
NEMOTRON_TEMPERATURE = 0.6

# --- Llama 3.2 Vision (Scene Analysis) ---
VISION_BASE_URL = "https://integrate.api.nvidia.com/v1"
VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
VISION_MAX_TOKENS = 512
VISION_TEMPERATURE = 0.2

# --- Riva ASR (Speech-to-Text) ---
RIVA_URI = "grpc.nvcf.nvidia.com:443"
RIVA_ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"  # Parakeet CTC 1.1B
RIVA_ASR_SAMPLE_RATE = 16000
RIVA_ASR_LANGUAGE = "en-US"

# --- Riva TTS (Text-to-Speech) ---
RIVA_TTS_FUNCTION_ID = "0149dedb-2be8-4195-b9a0-e57e0e14f972"  # FastPitch HiFi-GAN
RIVA_TTS_VOICE = "English-US.Female-1"
RIVA_TTS_SAMPLE_RATE = 44100
RIVA_TTS_LANGUAGE = "en-US"

# --- Webcam ---
WEBCAM_DEVICE_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 60

# --- Continuous Mode ---
CONTINUOUS_MODE_INTERVAL = 5  # seconds between frame captures
FRAME_DIFF_MSE_THRESHOLD = 500  # pixel MSE below this = scene unchanged
SCENE_CHANGE_THRESHOLD = 0.5  # word diff ratio above this = speak update
PROXIMITY_ALERT_FEET = 3  # distance threshold for proximity alerts

# --- Agent ---
AGENTIC_LOOP_MAX_ITERATIONS = 5

# --- Fallback flags ---
USE_RIVA_ASR = True
USE_RIVA_TTS = True

# --- Vision-only demo mode ---
VISION_ONLY_MODE = False
VISION_ONLY_MAX_TOKENS = 32           # 3-5 words ≈ 10-15 tokens
NEMOTRON_VISION_ONLY_MAX_TOKENS = 64  # short diff or NO_UPDATE
CONTINUOUS_MODE_INTERVAL_VISION_ONLY = 3  # faster loop for real-time
