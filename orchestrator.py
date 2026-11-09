"""
Central orchestrator: manages the state machine, conversation history,
tool dispatch, continuous mode, and live camera feed with debug overlay.

State Machine:
  LISTENING -> THINKING -> SPEAKING -> LISTENING
"""
import cv2
import datetime
import numpy as np
import os
import re
import threading
import time
from enum import Enum
import config
from agent import run_agentic_loop, create_initial_history
from speech import listen, speak, play_proximity_beep
from vision import capture_and_describe, capture_frame_raw, read_text, release_camera, get_camera
from navigation import get_directions
from prompts import CONTINUOUS_MODE_PROMPT_TEMPLATE

MAX_OVERLAY_LINES = 8
WRAP_WIDTH = 70


class AppState(Enum):
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Orchestrator:
    def __init__(self):
        self.state = AppState.LISTENING
        self.conversation_history = create_initial_history()
        self.continuous_mode = False
        self.continuous_timer = None
        self.continuous_interval = config.CONTINUOUS_MODE_INTERVAL
        self.last_scene_description = None
        self.running = True
        self._prev_frame = None

        # Thread-safe overlay log buffer
        self._overlay_lines = []
        self._overlay_lock = threading.Lock()


        # File-based debug log
        os.makedirs("logs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = open(f"logs/debug_{ts}.log", "w")

        self.tool_handlers = {
            "capture_and_describe": self._logged_capture_and_describe,
            "read_text": self._logged_read_text,
            "get_directions": self._logged_get_directions,
            "toggle_continuous_mode": self._handle_toggle_continuous,
        }

    def _log(self, tag, msg):
        """Debug log to terminal and overlay buffer."""
        ts = time.strftime("%H:%M:%S")
        full = f"[{ts}] [{tag}] {msg}"
        print(full)
        self._log_file.write(full + "\n")
        self._log_file.flush()

        # Truncate long messages for the overlay
        short = msg if len(msg) <= WRAP_WIDTH else msg[:WRAP_WIDTH] + "..."
        overlay_line = f"[{tag}] {short}"

        with self._overlay_lock:
            self._overlay_lines.append(overlay_line)
            # Keep only the most recent lines
            if len(self._overlay_lines) > MAX_OVERLAY_LINES:
                self._overlay_lines = self._overlay_lines[-MAX_OVERLAY_LINES:]

    def _logged_capture_and_describe(self, **kwargs):
        self._log("TOOL", f"capture_and_describe({kwargs})")
        result = capture_and_describe(**kwargs)
        self._log("VISION", result)
        return result

    def _logged_read_text(self, **kwargs):
        self._log("TOOL", "read_text()")
        result = read_text(**kwargs)
        self._log("VISION", result)
        return result

    def _logged_get_directions(self, **kwargs):
        self._log("TOOL", f"get_directions({kwargs})")
        result = get_directions(**kwargs)
        self._log("NAV", result)
        return result

    def _handle_toggle_continuous(self, enabled: bool, interval_seconds: float = None) -> str:
        if interval_seconds:
            self.continuous_interval = interval_seconds

        if enabled and not self.continuous_mode:
            self.continuous_mode = True
            self._start_continuous_timer()
            return f"Continuous mode enabled. I'll check your surroundings every {self.continuous_interval} seconds."
        elif not enabled and self.continuous_mode:
            self.continuous_mode = False
            self._stop_continuous_timer()
            return "Continuous mode disabled."
        elif enabled:
            return "Continuous mode is already on."
        else:
            return "Continuous mode is already off."

    def _start_continuous_timer(self):
        if self.continuous_timer:
            self.continuous_timer.cancel()

        def _continuous_loop():
            while self.continuous_mode and self.running:
                if self.state == AppState.LISTENING:
                    self._process_continuous_update()
                time.sleep(self.continuous_interval)

        self.continuous_timer = threading.Thread(target=_continuous_loop, daemon=True)
        self.continuous_timer.start()

    def _stop_continuous_timer(self):
        self.continuous_mode = False

    def _is_proximity_alert(self, text):
        """Check if response mentions an object within proximity threshold."""
        match = re.search(r'(\d+)\s*(?:feet|ft)', text, re.IGNORECASE)
        if match and int(match.group(1)) <= config.PROXIMITY_ALERT_FEET:
            return True
        return bool(re.search(r'\b(close|near|approaching|right here)\b', text, re.IGNORECASE))

    def _process_continuous_update(self):
        try:
            # Frame diff gate: skip API calls if scene barely changed
            if config.VISION_ONLY_MODE:
                frame = capture_frame_raw()
                if self._prev_frame is not None:
                    mse = np.mean((frame.astype(float) - self._prev_frame.astype(float)) ** 2)
                    if mse < config.FRAME_DIFF_MSE_THRESHOLD:
                        self._log("CONTINUOUS", f"Scene unchanged (MSE={mse:.0f}), skipping")
                        self._prev_frame = frame
                        return
                self._prev_frame = frame

            t0 = time.time()
            current_description = capture_and_describe()
            # Strip markdown formatting from vision output
            current_description = re.sub(r'\*+', '', current_description)
            current_description = re.sub(r'^#+\s*', '', current_description, flags=re.MULTILINE)
            current_description = current_description.strip()
            t1 = time.time()
            self._log("LATENCY", f"Vision: {t1 - t0:.1f}s")
            self._log("CONTINUOUS", current_description)

            if self.last_scene_description:
                # Skip Nemotron — compare vision descriptions directly for latency
                prev_words = set(self.last_scene_description.lower().split())
                curr_words = set(current_description.lower().split())
                diff_ratio = len(curr_words.symmetric_difference(prev_words)) / max(len(curr_words | prev_words), 1)

                if diff_ratio > config.SCENE_CHANGE_THRESHOLD:
                    self._log("CONTINUOUS", f"Scene changed (diff={diff_ratio:.0%}): {current_description}")
                    self.state = AppState.SPEAKING
                    if self._is_proximity_alert(current_description):
                        self._log("ALERT", f"Proximity alert: \"{current_description}\"")
                    t4 = time.time()
                    speak(current_description)
                    t5 = time.time()
                    self._log("LATENCY", f"TTS: {t5 - t4:.1f}s")
                    self._log("LATENCY", f"Total: {t5 - t0:.1f}s")
                    self.state = AppState.LISTENING
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": f"[Continuous mode alert] {current_description}"
                    })
                else:
                    self._log("CONTINUOUS", f"Text similar (diff={diff_ratio:.0%}), skipping")

            self.last_scene_description = current_description

        except Exception as e:
            self._log("ERROR", f"Continuous mode: {e}")

    def _draw_overlay(self, frame):
        """Draw state label and log lines on the camera frame."""
        h, w = frame.shape[:2]

        # --- State label at top-left (green) ---
        state_text = f"State: {self.state.value.upper()}"
        cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # --- Log overlay at bottom ---
        with self._overlay_lock:
            lines = list(self._overlay_lines)

        if not lines:
            return frame

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        line_height = 20
        padding = 8

        # Calculate overlay height
        overlay_h = len(lines) * line_height + padding * 2

        # Draw semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - overlay_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw each log line
        for i, line in enumerate(lines):
            y = h - overlay_h + padding + (i + 1) * line_height - 4
            cv2.putText(frame, line, (8, y), font, font_scale, (255, 255, 255), thickness)

        return frame

    def _speech_loop(self):
        """Background thread: listen → think → speak loop."""
        self._log("STATE", "Third Eye starting...")
        speak("Third Eye is ready. How can I help you?")

        while self.running:
            try:
                self.state = AppState.LISTENING
                self._log("STATE", "LISTENING — waiting for speech...")
                user_input = listen()

                if not user_input:
                    continue

                self._log("ASR", f"User said: \"{user_input}\"")

                if user_input.lower().strip() in ["exit", "quit", "stop", "goodbye", "bye"]:
                    speak("Goodbye! Stay safe.")
                    self.running = False
                    break

                self.state = AppState.THINKING
                self._log("STATE", "THINKING — sending to Nemotron...")
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input
                })

                response = run_agentic_loop(
                    self.conversation_history,
                    self.tool_handlers
                )

                self._log("NEMOTRON", f"Response: {response}")

                self.state = AppState.SPEAKING
                self._log("STATE", "SPEAKING...")
                speak(response)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                self._log("ERROR", f"Main loop: {e}")
                speak("I encountered an error. Let me try again.")

    def run(self):
        """Main thread runs camera feed (required by macOS), speech loop in background."""
        if config.VISION_ONLY_MODE:
            # Vision-only: no mic, auto-start continuous mode with faster interval
            self._log("STATE", "Vision-only mode starting...")
            speak("Watching for you.")
            self.continuous_mode = True
            self.continuous_interval = config.CONTINUOUS_MODE_INTERVAL_VISION_ONLY
            self._start_continuous_timer()
        else:
            # Normal mode: speech loop in background thread
            speech_thread = threading.Thread(target=self._speech_loop, daemon=True)
            speech_thread.start()

        # Camera feed on main thread (macOS requires cv2.imshow on main thread)
        camera = get_camera()
        while self.running:
            ret, frame = camera.read()
            if ret:
                frame = self._draw_overlay(frame)
                cv2.imshow("Third Eye - Live Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                self.running = False
                break

        cv2.destroyAllWindows()
        self._stop_continuous_timer()
        release_camera()
