import uuid
import time
import requests
from stt2 import transcribe_from_mic
from tts2 import speak_text

# RAG server URL (ensure this matches your server)
RAG_API_URL = "http://localhost:8000/ask"
session = requests.Session()
SESSION_ID = str(uuid.uuid4())  # keep same session id for the run (memory preserved)

def ask_rag(question: str) -> str:
    payload = {"session_id": SESSION_ID, "question": question}
    try:
        resp = session.post(RAG_API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("answer", "(no answer)")
    except Exception as e:
        print("Error calling RAG server:", e)
        return "(error contacting RAG service)"

def main_loop():
    print("=== HR Voice Assistant (Azure STT + TTS) ===")
    print("Press Ctrl+C to quit.\n")

    while True:
        try:
            # 1) STT: listen once (blocked until end-of-utterance)
            text = transcribe_from_mic(timeout_seconds=8)

            if not text:
                # nothing recognized — continue listening
                # small sleep avoids CPU spin
                time.sleep(0.2)
                continue

            print(f"\nYou said: {text}")

            # 2) Send text to RAG
            answer = ask_rag(text)
            print(f"\nAssistant: {answer}\n")

            # 3) Speak back the answer (TTS)
            speak_text(answer)

            # immediate re-listen (hot mic behavior)
            time.sleep(0.2)

        except KeyboardInterrupt:
            print("\nExiting voice loop.")
            break
        except Exception as e:
            print("Runtime error in voice loop:", e)
            time.sleep(1)

if __name__ == "__main__":
    main_loop()
