import tempfile
import sounddevice as sd
import soundfile as sf
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(api_version=os.environ["AZURE_OPENAI_API_VERSION"])

def speak_text(text: str):
    # Request streaming WAV audio
    response = client.audio.speech.with_streaming_response.create(
        model=os.getenv("TTS_MODEL"),
        voice="onyx", # Can choose other voices: alloy (serious female), onyx,(serious male) verse (cheerful male), nova (cheerful female), shimmer (serious female)
        input=text,
        response_format="wav",
    )

    # Save streamed bytes
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    # IMPORTANT: Use context manager correctly
    with response as stream:
        with open(temp.name, "wb") as f:
            for chunk in stream.iter_bytes():
                f.write(chunk)

    # Load WAV and play
    audio_data, samplerate = sf.read(temp.name)
    sd.play(audio_data, samplerate)
    sd.wait()

    return temp.name