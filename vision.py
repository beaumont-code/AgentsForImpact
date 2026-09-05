"""
Handles webcam frame capture via OpenCV and scene analysis via Llama 3.2 Vision API.
"""
import cv2
import base64
import time
from openai import OpenAI
import config
from prompts import (
    VISION_DESCRIBE_PROMPT, VISION_FOCUSED_PROMPT_TEMPLATE,
    VISION_READ_TEXT_PROMPT
)

# --- Vision API client ---
vision_client = OpenAI(
    base_url=config.VISION_BASE_URL,
    api_key=config.NVIDIA_API_KEY
)

# --- Webcam handle (kept open for low-latency repeated captures) ---
_camera = None
_last_raw_frame = None  # Most recent raw frame for live preview


def _get_camera():
    """Lazy-initialize and return the webcam handle."""
    global _camera
    if _camera is None or not _camera.isOpened():
        # CAP_AVFOUNDATION is the most reliable backend on macOS
        _camera = cv2.VideoCapture(config.WEBCAM_DEVICE_INDEX, cv2.CAP_AVFOUNDATION)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not _camera.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check System Settings > Privacy & Security > Camera."
            )
        time.sleep(0.5)  # macOS camera needs a moment to warm up
    return _camera


def get_camera():
    """Public access to the shared camera handle."""
    return _get_camera()


def capture_frame() -> bytes:
    """
    Capture a single frame from the webcam and return it as JPEG bytes.

    Returns:
        JPEG-encoded image bytes, resized to FRAME_WIDTH x FRAME_HEIGHT.
    """
    global _last_raw_frame
    camera = _get_camera()
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from webcam.")

    _last_raw_frame = frame.copy()
    frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
    _, jpeg_bytes = cv2.imencode('.jpg', frame, encode_params)
    return jpeg_bytes.tobytes()


def capture_frame_from_file(path: str) -> bytes:
    """Load a test photo as JPEG bytes (for development without webcam)."""
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Could not read test image: {path}")
    frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    _, jpeg_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return jpeg_bytes.tobytes()


def analyze_frame(jpeg_bytes: bytes, prompt: str) -> str:
    """
    Send a JPEG frame to Llama 3.2 Vision API and get a text description.

    Args:
        jpeg_bytes: JPEG-encoded image bytes.
        prompt: The text prompt to send alongside the image.

    Returns:
        Text description from the vision model.
    """
    b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")

    response = vision_client.chat.completions.create(
        model=config.VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=config.VISION_ONLY_MAX_TOKENS if config.VISION_ONLY_MODE else config.VISION_MAX_TOKENS,
        temperature=config.VISION_TEMPERATURE,
    )
    return response.choices[0].message.content


def capture_and_describe(focus: str = None) -> str:
    """
    Tool handler: capture a webcam frame and describe the scene.

    Args:
        focus: Optional focus area (e.g., "obstacles", "people", "traffic").
    """
    frame = capture_frame()
    if focus:
        prompt = VISION_FOCUSED_PROMPT_TEMPLATE.format(focus=focus)
    else:
        prompt = VISION_DESCRIBE_PROMPT
    return analyze_frame(frame, prompt)


def read_text() -> str:
    """Tool handler: capture a webcam frame and read all visible text."""
    frame = capture_frame()
    return analyze_frame(frame, VISION_READ_TEXT_PROMPT)


def capture_frame_raw():
    """Capture a frame and return raw numpy array for frame comparison."""
    capture_frame()
    return _last_raw_frame


def release_camera():
    """Release the webcam handle. Call on shutdown."""
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None
