"""
Handbook Retrieval Tool
LangChain tool for searching the employee handbook vectorstore.
"""
from langchain_core.tools import tool
from app.repositories.handbook_repository import retriever


@tool
def handbook_retriever_tool(query: str) -> str:
    """
    Search the Employee Handbook for policies, procedures, definitions, entitlements, and rules.
    Returns the top relevant sections.
    """
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the employee handbook."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)