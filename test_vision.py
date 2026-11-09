"""
Live vision test: shows webcam feed and sends frames to Llama Vision API on keypress.

Controls:
    d - Describe scene
    r - Read text
    q / ESC - Quit

Usage:
    python test_vision.py
"""
import cv2
import time
from vision import analyze_frame
from prompts import VISION_DESCRIBE_PROMPT, VISION_READ_TEXT_PROMPT
from config import WEBCAM_DEVICE_INDEX, FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY

cap = cv2.VideoCapture(WEBCAM_DEVICE_INDEX, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

time.sleep(0.5)  # macOS camera warm-up

print("Live Vision Test")
print("  d = describe scene")
print("  r = read text")
print("  q / ESC = quit")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    cv2.imshow("Third Eye - Live Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

    if key in (ord("d"), ord("r")):
        # Encode current frame as JPEG
        frame_resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        _, jpeg_bytes = cv2.imencode('.jpg', frame_resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpeg = jpeg_bytes.tobytes()

        if key == ord("d"):
            prompt = VISION_DESCRIBE_PROMPT
            label = "SCENE DESCRIPTION"
        else:
            prompt = VISION_READ_TEXT_PROMPT
            label = "TEXT READING"

        print(f"\n{'=' * 50}")
        print(f"{label} (sending {len(jpeg)} bytes...)")
        print(f"{'=' * 50}")

        try:
            start = time.time()
            result = analyze_frame(jpeg, prompt)
            elapsed = time.time() - start
            print(result)
            print(f"\n[{elapsed:.1f}s response time]")
        except Exception as e:
            print(f"API error: {e}")

cap.release()
cv2.destroyAllWindows()
