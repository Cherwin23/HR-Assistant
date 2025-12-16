# HR Voice Assistant

HR Voice Assistant with Intent Classification and RAG (Retrieval-Augmented Generation).

## Requirements
- Python 3.12+
- Install dependencies: `pip install -r requirements.txt`

## Project Structure
- **Controllers**: HTTP request handlers (`app/controllers/`)
- **Services**: Business logic (`app/services/`)
- **Repositories**: Data access layer (`app/repositories/`)
- **Agents**: LangGraph agents (`app/agents/`)
- **Tools**: LangChain tools (`app/tools/`)
- **Models**: Data models (`app/models/`)

## Quick Start

### 1. Build Vectorstore (One-time setup)
```bash
python app/utils/build_vectorstore.py
```

### 2. Start FastAPI Server
```bash
uvicorn app.main:app --reload
```

### 3. Run Voice Assistant

**Option 1: OpenAI STT/TTS**
```bash
python app/utils/voice_loop.py
```

**Option 2: Azure Speech SDK**
```bash
python app/utils/voice_loop2.py
```