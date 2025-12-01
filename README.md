## Requirements
- Python 3.12+
- Install dependencies with:
  pip install -r requirements.txt

# Option 1: Using GPT models
  * Run voice_loop.py (stt.py -> rag.py -> tts.py)

# Option 2: Using Azure Speech Service
  * Run voice_loop2.py (stt2.py -> rag.py -> tts2.py)

# Steps to run
1. Run build_vectorstore.py to create Chroma Vectorstore (Only once)
2. Run FastAPI for RAG
  * Open terminal, activate virtual env, navigate to project folder, run uvicorn app.server:app --reload
3. Run voice_loop.py or voice_loop2.py
