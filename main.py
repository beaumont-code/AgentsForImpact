"""
Third Eye - Entry point.
An AI navigation and surroundings assistant for visually impaired people.

Usage:
    python main.py                  # Normal mode (Riva speech)
    python main.py --no-riva        # Use fallback ASR/TTS (no Riva)
"""
import argparse
import config
from orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Third Eye - AI Vision Assistant")
    parser.add_argument(
        "--no-riva",
        action="store_true",
        help="Use fallback speech engines instead of NVIDIA Riva"
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Vision-only demo mode: no mic, auto continuous capture + TTS alerts"
    )
    args = parser.parse_args()

    if args.no_riva:
        config.USE_RIVA_ASR = False
        config.USE_RIVA_TTS = False
        print("Using fallback speech engines (SpeechRecognition + pyttsx3)")

    if args.vision_only:
        config.VISION_ONLY_MODE = True
        config.USE_RIVA_ASR = False
        print("Vision-only mode: mic disabled, continuous vision active")

    print("=" * 50)
    print("  THIRD EYE - AI Vision Assistant")
    if config.VISION_ONLY_MODE:
        print("  VISION-ONLY MODE — auto continuous capture")
    else:
        print("  Speak to navigate. Say 'exit' to quit.")
    print("=" * 50)
    print()

    orchestrator = Orchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
