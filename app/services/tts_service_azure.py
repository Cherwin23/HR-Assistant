"""
Text-to-Speech Service (Azure Speech SDK)
Handles text-to-speech synthesis using Azure Speech SDK.
"""
import os
import re
import tempfile
import time
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import sounddevice as sd
import soundfile as sf
from app.config.settings import AZURE_SPEECH_KEY, AZURE_SPEECH_REGION

load_dotenv()

# Azure Speech SDK Config
speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY,
    region=AZURE_SPEECH_REGION
)

# Voice Options: en-SG-LunaNeural (Female Singlish), en-SG-WayneNeural (Male Singlish), en-US-JennyNeural (Female English)
speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"


def prepare_text_for_speech(text: str) -> str:
    """
    Converts markdown-heavy text into natural speech.
    Removes symbols and rewrites common patterns.
    
    Args:
        text: Text to prepare for speech
        
    Returns:
        Cleaned text suitable for TTS
    """
    # Remove markdown headers ###, ##, #
    text = re.sub(r"#+\s*", "", text)

    # Replace bullets
    text = re.sub(r"[-•]\s*", " - ", text)

    # Replace "1st", "2nd", "3rd", "4th" etc.
    text = re.sub(r"\b1st\b", "first", text, flags=re.IGNORECASE)
    text = re.sub(r"\b2nd\b", "second", text, flags=re.IGNORECASE)
    text = re.sub(r"\b3rd\b", "third", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+)th\b", r"\1th", text)  # optional

    # Remove bold/italic markdown
    text = text.replace("**", "")
    text = text.replace("*", "")

    # Remove code ticks
    text = text.replace("`", "")

    # Remove excessive parentheses for citations
    text = text.replace("(see:", "See")
    text = text.replace(")", "")

    # Convert multiple newlines to pauses
    text = re.sub(r"\n{2,}", ". ", text)
    text = text.replace("\n", ", ")

    return text.strip()


def speak_text(text: str) -> None:
    """
    Synthesize and speak text using Azure Speech SDK.
    
    Args:
        text: Text to speak (will be cleaned automatically)
    """
    spoken_text = prepare_text_for_speech(text)

    # 1) Create temp WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav_path = tmp.name

    # 2) Synthesize to WAV file
    audio_cfg = speechsdk.audio.AudioOutputConfig(filename=wav_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_cfg)
    result = synthesizer.speak_text_async(spoken_text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("TTS failed:", result.reason)
        return

    # 3) Ensure file is done
    time.sleep(0.05)

    # 4) Load the file safely
    data, samplerate = sf.read(wav_path)

    # 5) Play via sounddevice
    sd.play(data, samplerate)
    sd.wait()

    # Cleanup
    os.remove(wav_path)

