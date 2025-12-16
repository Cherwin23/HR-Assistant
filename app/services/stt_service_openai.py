"""
Speech-to-Text Service (OpenAI)
Handles audio transcription using OpenAI Whisper via Azure OpenAI.
Includes hot mic recording with VAD.
"""
import os
import tempfile
import wave
import time
import webrtcvad
import collections
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# OpenAI STT Client
stt_client = AzureOpenAI(api_version=os.environ["AZURE_OPENAI_API_VERSION"])

# VAD HOT MIC SETTINGS
RATE = 16000
CHANNELS = 1
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # samples per frame
VAD = webrtcvad.Vad(2)  # 0=least aggressive, 3=most aggressive


def is_speech(frame_bytes) -> bool:
    """Return True if frame contains speech."""
    return VAD.is_speech(frame_bytes, RATE)


def record_hot_mic() -> str:
    """
    Continuously listens, detects voice activity,
    records until silence, and returns a WAV temp file path.
    Uses PyAudio and WebRTC VAD.
    """
    import pyaudio
    
    print("🎤 Hot mic listening... start speaking!")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE,
    )

    # ring buffers for smooth detection
    voiced_frames = []
    ring_buffer = collections.deque(maxlen=10)

    speaking = False
    start_time = None

    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)

        if is_speech(frame):
            ring_buffer.append(True)
        else:
            ring_buffer.append(False)

        # Speech START detected
        if not speaking and sum(ring_buffer) > 7:
            speaking = True
            start_time = time.time()
            print("🟢 Speech detected. Recording...")
            voiced_frames = []

        # Capture frames while speaking
        if speaking:
            voiced_frames.append(frame)

            # Silence END detected
            if sum(ring_buffer) < 2 and (time.time() - start_time) > 0.6:
                print("🔴 Silence detected. Stopped recording.")
                break

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Save to temp wav
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(RATE)
        wf.writeframes(b"".join(voiced_frames))

    return temp_file.name


def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper (Azure OpenAI).
    
    Args:
        file_path: Path to audio file (WAV format)
        
    Returns:
        Transcribed text
    """
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    resp = stt_client.audio.transcriptions.create(
        model=os.getenv("STT_MODEL"),
        file=("speech.wav", audio_bytes, "audio/wav"),
        language="en",
        temperature=0.2,
    )
    return resp.text.strip()

