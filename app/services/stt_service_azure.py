"""
Speech-to-Text Service (Azure Speech SDK)
Handles audio transcription using Azure Speech SDK.
"""
import os
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from app.config.settings import AZURE_SPEECH_KEY, AZURE_SPEECH_REGION

load_dotenv()

# Azure Speech SDK Config
speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY,
    region=AZURE_SPEECH_REGION
)

# Force English (adjust to en-SG or en-US as you prefer)
speech_config.speech_recognition_language = "en-SG"


def transcribe_from_mic(timeout_seconds: int = 8) -> str:
    """
    Listens to the default microphone and returns a single-utterance transcription.
    Azure's recognizer handles the voice activity detection / end-of-utterance detection.
    
    Args:
        timeout_seconds: maximum seconds to wait for the utterance (approx).
        
    Returns:
        Transcribed text, or empty string if no speech detected
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

