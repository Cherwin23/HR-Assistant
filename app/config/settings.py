"""
Application configuration and settings.
Centralized environment variables and constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
# ENV controls environment-specific behavior:
# - dev  : local development (CSV-backed employee DB, local Chroma vectorstore)
# - uat  : staging / UAT (pre-provisioned employee DB, future Azure AI Search)
# - prod : production     (pre-provisioned employee DB, future Azure AI Search)
ENV = os.getenv("ENV", "dev").lower()

# -----------------------------------------------------------------------------
# Azure OpenAI Configuration
# -----------------------------------------------------------------------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
STT_MODEL = os.getenv("STT_MODEL")
TTS_MODEL = os.getenv("TTS_MODEL")
# Deployment name for Realtime API (To Implement)
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime")

# -----------------------------------------------------------------------------
# Azure Speech Service
# -----------------------------------------------------------------------------
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

# -----------------------------------------------------------------------------
# Vector Store Configuration
# -----------------------------------------------------------------------------
# DEV  : Local Chroma (directory on disk)
# UAT/PROD: Will use Azure AI Search (configured via AZURE_SEARCH_*), but for now
#           we still point to Chroma so the app continues to function.

# Local Chroma configuration
VECTORSTORE_PERSIST_DIRECTORY = os.getenv("VECTORSTORE_PERSIST_DIRECTORY")
VECTORSTORE_COLLECTION_NAME = os.getenv("VECTORSTORE_COLLECTION_NAME")

# Azure AI Search placeholders (for future use in UAT/PROD)
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")  # e.g. https://<search>.search.windows.net
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")  # e.g. hr-rag-index
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

# -----------------------------------------------------------------------------
# Employee Data Configuration
# -----------------------------------------------------------------------------
# DEV  : CSV is the source of truth; SQLite DB is auto-built from CSV if missing.
# UAT/PROD: Expect a pre-provisioned SQLite DB; CSV is not used to auto-build DB.

EMPLOYEE_CSV_PATH = os.getenv("CSV_PATH")  # path to CSV used in dev to seed DB

if ENV == "dev":
    # Local dev DB path (auto-created from CSV if missing)
    EMPLOYEE_DB_PATH = os.getenv("DB_PATH", "employee_data.db")
else:
    # UAT / PROD: pre-provisioned DB path; must exist, no auto-build from CSV
    EMPLOYEE_DB_PATH = os.getenv("DB_PATH", "employee_data_prod.db")

# -----------------------------------------------------------------------------
# PDF Configuration
# -----------------------------------------------------------------------------
PDF_PATH = os.getenv("PDF_PATH")

# -----------------------------------------------------------------------------
# Summarization Configuration (Default to 100 words)
# -----------------------------------------------------------------------------
SUMMARY_MAX_WORDS = int(os.getenv("SUMMARY_MAX_WORDS", "100"))

# -----------------------------------------------------------------------------
# Azure Blob Storage Configuration (for audit trails)
# -----------------------------------------------------------------------------
# Provide either connection_string OR (account_name + account_key)
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "hr-assistant-audit")

# -----------------------------------------------------------------------------
# Prompt Paths
# -----------------------------------------------------------------------------
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
INTENT_CLASSIFICATION_PROMPT_PATH = "prompts/intent_classification_prompt.txt"
SUMMARIZATION_PROMPT_PATH = "prompts/summarization_prompt.txt"
