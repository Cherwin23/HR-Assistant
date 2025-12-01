import os
import uuid
import requests
from dotenv import load_dotenv

from stt import record_hot_mic, transcribe_audio
from tts import speak_text

load_dotenv()

RAG_API_URL = "http://localhost:8000/ask"
SESSION_ID = str(uuid.uuid4())    # persistent conversation

def ask_rag(question: str):
    """Send question to your FastAPI RAG server."""
    payload = {"session_id": SESSION_ID, "question": question}
    resp = requests.post(RAG_API_URL, json=payload)
    return resp.json().get("answer", "(no answer)")

def voice_loop():
    """Full voice → STT → RAG → TTS assistant loop."""
    while True:
        print("\n🎤 Speak now…")

        wav_file = record_hot_mic()         # Hot mic listening
        text = transcribe_audio(wav_file)   # Azure GPT-4o-mini transcription
        os.remove(wav_file)

        if not text.strip():
            print("❗ No speech detected.")
            continue

        print(f"\n🗣️ You said: {text}")

        rag_answer = ask_rag(text)
        print(f"\n🤖 HR Assistant: {rag_answer}")

        speak_text(rag_answer)


if __name__ == "__main__":
    voice_loop()
