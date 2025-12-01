import os
import re
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

speech_config = speechsdk.SpeechConfig(
    subscription=os.getenv("AZURE_SPEECH_KEY"),
    region=os.getenv("AZURE_SPEECH_REGION")
)

# Recommended Singapore voice; change if you prefer another voice
#speech_config.speech_synthesis_voice_name = "en-SG-LunaNeural"
speech_config.speech_synthesis_voice_name = "en-SG-WayneNeural"

audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

def prepare_text_for_speech(text: str) -> str:
    """
    Converts markdown-heavy text into natural speech.
    Removes symbols and rewrites common patterns.
    """

    # Remove markdown headers ###, ##, #
    text = re.sub(r"#+\s*", "", text)

    # Replace bullets
    text = re.sub(r"[-•]\s*", " - ", text)

    # Replace “1st”, “2nd”, “3rd”, “4th” etc.
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


def speak_text(text: str):
    """Speaks cleaned text using Azure TTS."""
    spoken_text = prepare_text_for_speech(text)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # Correct method — async + block until finished
    result = synthesizer.speak_text_async(spoken_text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("TTS failed:", result.reason)