import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

# Create and reuse speech config (singleton)
speech_config = speechsdk.SpeechConfig(
    subscription=os.getenv("AZURE_SPEECH_KEY"),
    region=os.getenv("AZURE_SPEECH_REGION")
)
# Force English (adjust to en-SG or en-US as you prefer)
speech_config.speech_recognition_language = "en-US"

# Use the default system microphone
def transcribe_from_mic(timeout_seconds: int = 8) -> str:
    """
    Listens to the default microphone and returns a single-utterance transcription.
    Azure's recognizer handles the voice activity detection / end-of-utterance detection.
    timeout_seconds: maximum seconds to wait for the utterance (approx).
    """
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    try:
        print("Listening (speak now)...")
        # recognize_once_async listens until it detects end of utterance or timeout
        result = recognizer.recognize_once()
    except Exception as e:
        print("STT error / timeout:", e)
        return ""

    if result is None:
        return ""

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = result.text.strip()
        return text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        # no speech recognized
        return ""
    else:
        # canceled or error
        print("Recognition canceled/failed:", result.reason)
        return ""
