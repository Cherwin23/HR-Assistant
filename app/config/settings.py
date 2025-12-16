"""
Application configuration and settings.
Centralized environment variables and constants.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
STT_MODEL = os.getenv("STT_MODEL")
TTS_MODEL = os.getenv("TTS_MODEL")

# Azure Speech Service (STT + TTS)
AZURE_SPEECH_KEY=os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION=os.getenv("AZURE_SPEECH_REGION")

# Vector Store Configuration
VECTORSTORE_PERSIST_DIRECTORY = "chroma_langchain_db"
VECTORSTORE_COLLECTION_NAME = "hr_rag"

# Employee Data Configuration
EMPLOYEE_CSV_PATH = os.getenv("CSV_PATH")
EMPLOYEE_DB_PATH = os.getenv("DB_PATH", "employee_data.db") # to set in .env

# PDF Configuration
PDF_PATH = os.getenv("PDF_PATH")

# Prompt Paths
SYSTEM_PROMPT_PATH = "prompts/system_prompt.txt"
INTENT_CLASSIFICATION_PROMPT_PATH = "prompts/intent_classification_prompt.txt"