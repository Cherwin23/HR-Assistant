"""
Handbook Repository
Handles employee handbook vectorstore operations.

Environment behavior:
- All environments currently use a local Chroma vectorstore on disk, using
  `VECTORSTORE_PERSIST_DIRECTORY` and `VECTORSTORE_COLLECTION_NAME` from
  `app.config.settings`.
- In the future, UAT/PROD can be switched to an external vector service
  (e.g. Azure AI Search) by updating this module while keeping the retriever
  interface the same.
"""
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from app.config.settings import VECTORSTORE_PERSIST_DIRECTORY, VECTORSTORE_COLLECTION_NAME, EMBEDDING_MODEL

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL)

# Initialize local Chroma vectorstore.
# Note: Even in UAT/PROD we currently use Chroma; once Azure AI Search is wired up,
# this module can branch on ENV to use the remote vector index instead.
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=VECTORSTORE_PERSIST_DIRECTORY,
    collection_name=VECTORSTORE_COLLECTION_NAME
)

# Create retriever (same behavior across environments for now)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)