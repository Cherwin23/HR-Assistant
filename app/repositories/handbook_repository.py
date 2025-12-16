"""
Handbook Repository
Handles employee handbook vectorstore operations.
"""
import os
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from app.config.settings import (
    VECTORSTORE_PERSIST_DIRECTORY,
    VECTORSTORE_COLLECTION_NAME,
    EMBEDDING_MODEL
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(model=EMBEDDING_MODEL)

# Initialize vectorstore
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=VECTORSTORE_PERSIST_DIRECTORY,
    collection_name=VECTORSTORE_COLLECTION_NAME
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)