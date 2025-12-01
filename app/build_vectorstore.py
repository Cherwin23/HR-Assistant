from dotenv import load_dotenv
import os
import re
from typing import List
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

pdf_path = os.getenv("PDF_PATH")
persist_directory = "chroma_langchain_db"
collection_name = "hr_rag"

print(pdf_path)

# 1. Load PDF
# Safety measure for debugging purposes, Checks if the PDF is there
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()
print(f"[RAG] Loaded PDF ({len(pages)} pages)")

# 2. Header detection
# Combine pages into single text for header detection
full_text = "\n".join([p.page_content for p in pages])

TOP_LEVEL_RE = re.compile(r"^\s*\d+\s+[A-Z][A-Z\s]+(?:\d+)?\s*$")
SUB_LEVEL_RE = re.compile(r"^\s*\d+\.\d+\s+[A-Za-z].+")

def split_with_headers(text: str) -> List[Document]:
    """Split PDF text into documents with section + subsection metadata."""
    lines = text.splitlines()
    docs: List[Document] = []
    current_section = None
    current_subsection = None
    buffer_lines = []

    def flush():
        if buffer_lines:
            content = "\n".join(buffer_lines).strip()
            if content:
                metadata = {}
                if current_section:
                    metadata["section"] = current_section
                if current_subsection:
                    metadata["subsection"] = current_subsection
                docs.append(Document(page_content=content, metadata=metadata))
        buffer_lines.clear()

    for line in lines:
        stripped = line.strip()

        if TOP_LEVEL_RE.match(stripped):
            flush()
            current_section = " ".join(stripped.split())
            current_subsection = None
            buffer_lines.append(stripped)
            continue

        if SUB_LEVEL_RE.match(stripped):
            flush()
            current_subsection = " ".join(stripped.split())
            buffer_lines.append(stripped)
            continue

        buffer_lines.append(line)

    flush()
    return docs

header_docs = split_with_headers(full_text)
print(f"[RAG] Header-aligned documents: {len(header_docs)}")

# 3. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(header_docs)
print(f"[RAG] Total chunks created: {len(chunks)}")

# 4. Embedding + Vector Store
embeddings = AzureOpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name,
)

print("Vectorstore built successfully.")