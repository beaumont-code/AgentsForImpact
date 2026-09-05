"""
Voice I/O module wrapping NVIDIA Riva ASR (speech-to-text) and TTS (text-to-speech).
Includes fallbacks to SpeechRecognition + pyttsx3 if Riva is unavailable.
"""
import numpy as np
import sounddevice as sd
import config
from config import (
    NVIDIA_API_KEY, RIVA_URI,
    RIVA_ASR_FUNCTION_ID, RIVA_ASR_SAMPLE_RATE, RIVA_ASR_LANGUAGE,
    RIVA_TTS_FUNCTION_ID, RIVA_TTS_VOICE, RIVA_TTS_SAMPLE_RATE, RIVA_TTS_LANGUAGE,
)

_asr_service = None
_tts_service = None


def _init_riva_asr():
    """Initialize Riva ASR gRPC client."""
    global _asr_service
    import riva.client
    auth = riva.client.Auth(
        ssl_root_cert=None,
        use_ssl=True,
        uri=RIVA_URI,
        metadata_args=[
            ("function-id", RIVA_ASR_FUNCTION_ID),
            ("authorization", f"Bearer {NVIDIA_API_KEY}"),
        ]
    )
    _asr_service = riva.client.ASRService(auth)
    return _asr_service


def _init_riva_tts():
    """Initialize Riva TTS gRPC client."""
    global _tts_service
    import riva.client
    auth = riva.client.Auth(
        ssl_root_cert=None,
        use_ssl=True,
        uri=RIVA_URI,
        metadata_args=[
            ("function-id", RIVA_TTS_FUNCTION_ID),
            ("authorization", f"Bearer {NVIDIA_API_KEY}"),
        ]
    )
    _tts_service = riva.client.SpeechSynthesisService(auth)
    return _tts_service


def listen() -> str:
    """
    Record audio from the microphone and return the transcript.
    Uses Riva ASR if USE_RIVA_ASR is True, otherwise falls back to SpeechRecognition.
    """
    if config.USE_RIVA_ASR:
        return _listen_riva()
    else:
        return _listen_fallback()


def _listen_riva() -> str:
    """Record audio and transcribe using Riva ASR."""
    import riva.client

    if _asr_service is None:
        _init_riva_asr()

    duration_seconds = 10
    print("Listening... (speak now)")
    audio_data = sd.rec(
        int(duration_seconds * RIVA_ASR_SAMPLE_RATE),
        samplerate=RIVA_ASR_SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )

    input("Press Enter when done speaking...")
    sd.stop()

    audio_bytes = audio_data.tobytes()

    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        language_code=RIVA_ASR_LANGUAGE,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        sample_rate_hertz=RIVA_ASR_SAMPLE_RATE,
        audio_channel_count=1,
    )

    response = _asr_service.offline_recognize(audio_bytes, config)

    if response.results:
        transcript = response.results[0].alternatives[0].transcript
        print(f"You said: {transcript}")
        return transcript
    return ""


def _listen_fallback() -> str:
    """Fallback: use SpeechRecognition with Google's free API."""
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        print("Listening... (speak now)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)

    try:
        transcript = recognizer.recognize_google(audio)
        print(f"You said: {transcript}")
        return transcript
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"ASR error: {e}")
        return ""


def play_proximity_beep():
    """Play a short urgent beep for proximity alerts."""
    print("[BEEP] Playing proximity beep now")
    duration = 0.3
    freq = 1000
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    sd.play(tone, samplerate=sample_rate)
    sd.wait()


def speak(text: str) -> None:
    """
    Convert text to speech and play through speakers.
    Uses Riva TTS if USE_RIVA_TTS is True, otherwise falls back to pyttsx3.
    """
    if not text or text.strip() == "":
        return

    if config.USE_RIVA_TTS:
        _speak_riva(text)
    else:
        _speak_fallback(text)


def _speak_riva(text: str) -> None:
    """Synthesize and play speech using Riva TTS."""
    import riva.client

    if _tts_service is None:
        _init_riva_tts()

    response = _tts_service.synthesize(
        text=text,
        voice_name=RIVA_TTS_VOICE,
        language_code=RIVA_TTS_LANGUAGE,
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=RIVA_TTS_SAMPLE_RATE,
    )

    audio_array = np.frombuffer(response.audio, dtype=np.int16)
    sd.play(audio_array, samplerate=RIVA_TTS_SAMPLE_RATE)
    sd.wait()


def _speak_fallback(text: str) -> None:
    """Fallback: use pyttsx3 (macOS native TTS)."""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()
