# HR Voice Assistant

HR Voice Assistant with Intent Classification and RAG (Retrieval-Augmented Generation).

## Requirements
- Python 3.12+
- Install dependencies: `pip install -r requirements.txt`

## Environments

This project supports three environments, controlled by the `ENV` variable:

- `dev`  – local development  
- `uat`  – staging / UAT  
- `prod` – production  

Environment-specific behavior:

- **Employee data**
  - `dev`  : Uses `CSV_PATH` as source of truth and auto-builds a local SQLite DB at `DB_PATH` (default: `employee_data.db`) if it does not exist.
  - `uat`/`prod` : Expect a pre-provisioned SQLite DB at `DB_PATH` (default: `employee_data_prod.db`). The DB is **not** auto-created from CSV.
- **Vector store**
  - All envs currently use local Chroma in `VECTORSTORE_PERSIST_DIRECTORY` (e.g. `chroma_langchain_db`) with collection name `VECTORSTORE_COLLECTION_NAME`.
  - UAT/Prod are designed so they can later point to Azure AI Search (`AZURE_SEARCH_*` settings in `app.config.settings`).

You configure each environment via a corresponding `.env` file or deployment settings:

- `.env.dev`  → `ENV=dev`
- `.env.uat`  → `ENV=uat`
- `.env.prod` → `ENV=prod`

## Project Structure
- **Controllers**: HTTP request handlers (`app/controllers/`)
- **Services**: Business logic (`app/services/`)
- **Repositories**: Data access layer (`app/repositories/`)
- **Agents**: LangGraph agents (`app/agents/`)
- **Tools**: LangChain tools (`app/tools/`)
- **Models**: Data models (`app/models/`)

## Quick Start

### 1. Build Vectorstore (per environment)

The vectorstore is **derived data** and can be rebuilt from the source PDF.  
Run this after setting the appropriate `ENV` and `.env.*` file:

```bash
ENV=dev  python -m app.utils.build_vectorstore   # dev
ENV=uat  python -m app.utils.build_vectorstore   # uat (if using local Chroma)
ENV=prod python -m app.utils.build_vectorstore   # prod (if using local Chroma)
```

### 2. Start FastAPI Server
```bash
ENV=dev uvicorn app.main:app --reload
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