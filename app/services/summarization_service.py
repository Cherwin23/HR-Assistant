"""
Summarization Service
Generates concise summaries of RAG responses for voice interfaces.
"""
import asyncio
from typing import Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.config.settings import (
    CHAT_MODEL,
    AZURE_OPENAI_API_VERSION,
    SUMMARY_MAX_WORDS,
    SUMMARIZATION_PROMPT_PATH,
)
from app.utils.prompt_loader import load_prompt


# Initialize LLM for summarization
_summarization_llm = AzureChatOpenAI(
    model=CHAT_MODEL,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.3,  # Lower temperature for more consistent summaries
)


async def generate_summary(
    full_text: str, 
    max_words: Optional[int] = None
) -> str:
    """
    Generate a concise summary of the full response text.
    
    Args:
        full_text: The full RAG response text to summarize
        max_words: Maximum word count for summary (defaults to SUMMARY_MAX_WORDS from config)
    
    Returns:
        Concise summary string, or original text if summarization fails
    """
    if not full_text or not full_text.strip():
        return full_text
    
    # Use config default if not specified
    if max_words is None:
        max_words = SUMMARY_MAX_WORDS
    
    # If text is already short, no need to summarize
    word_count = len(full_text.split())
    if word_count <= max_words:
        return full_text
    
    # Load summarization prompt template and format with max_words
    prompt_template = load_prompt(SUMMARIZATION_PROMPT_PATH)
    system_prompt = prompt_template.format(max_words=max_words)
    
    user_prompt = f"Summarize the following text to approximately {max_words} words:\n\n{full_text}"
    
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await _summarization_llm.ainvoke(messages)
        summary = response.content.strip()
        
        # Fallback to original if summary is suspiciously short or empty
        if not summary or len(summary.split()) < 10:
            return full_text
        
        return summary
        
    except Exception as e:
        print(f"Summarization error: {e}")
        # Graceful fallback: return original text
        return full_text

