"""
RAG Service
Business logic for RAG processing with intent classification routing.
Includes summarization and audit trail storage for non-invalid queries.
"""
import time
import asyncio
from typing import Dict, Any, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from app.services.intent_service import classify_intent
from app.services.summarization_service import generate_summary
from app.services.blob_storage_service import store_interaction
from app.config.settings import SUMMARY_MAX_WORDS
from app.agents.rag_agent import rag_agent


async def process_with_intent_classification(
    question: str,
    conversation_history: Optional[List[BaseMessage]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process user question with intent classification and conditional RAG.
    Returns response matching schema.
    
    Args:
        question: User's input question
        conversation_history: Optional list of previous messages for context
        session_id: Optional session ID for audit trail storage
    
    Returns:
        Dictionary matching response schema:
        {
            "intent": str,
            "category": str,
            "module": str | None,
            "use_case": str | None,
            "answer": str | None,
            "confidence": float,
            "requires_context": List[str],
            "entities": Dict[str, Any]
        }
    """

    start_time = time.time()
    
    # Step 1: Intent Classification
    intent_result = await classify_intent(question, conversation_history)
    
    category = intent_result.get("category", "").lower()
    confidence = intent_result.get("confidence", 0.0)
    
    # Step 2: Category-Based Routing
    
    # INVALID or low confidence: Return rejection (skip storage)
    if category == "invalid" or confidence < 0.6:
        return {
            "intent": intent_result.get("intent", "invalid"),
            "category": "invalid",
            "module": intent_result.get("module"),
            "use_case": intent_result.get("use_case"),
            "answer": intent_result.get("answer", "I can only assist with HR-related queries. Please rephrase your question."),
            "confidence": confidence,
            "requires_context": [],
            "entities": intent_result.get("entities", {})
        }
    
    # CONVERSATIONAL: Return pre-generated answer
    if category == "conversational":
        return {
            "intent": intent_result.get("intent", "conversational"),
            "category": "conversational",
            "module": None,
            "use_case": None,
            "answer": intent_result.get("answer", "Hello! How can I help you with HR-related questions today?"),
            "confidence": confidence,
            "requires_context": [],
            "entities": {}
        }
    
    # ACTION: Return intent classification result (answer will be null, Gateway handles it)
    if category == "action":
        response_time_ms = (time.time() - start_time) * 1000
        
        # Store action interactions too (non-invalid)
        if session_id:
            asyncio.create_task(
                store_interaction(
                    session_id=session_id,
                    question=question,
                    intent_result=intent_result,
                    full_response="",  # No answer for actions
                    summary="",
                    summary_length=0,
                    tools_used=None,
                    response_time_ms=response_time_ms,
                )
            )
        
        return {
            "intent": intent_result.get("intent"),
            "category": "action",
            "module": intent_result.get("module"),
            "use_case": intent_result.get("use_case"),
            "answer": None,  # Gateway will handle action processing
            "confidence": confidence,
            "requires_context": intent_result.get("requires_context", []),
            "entities": intent_result.get("entities", {})
        }
    
    # QUERY: Use RAG to generate answer
    if category == "query":
        # Prepare messages for RAG
        if conversation_history:
            rag_messages = list(conversation_history)
        else:
            rag_messages = []
        
        # Add current question
        rag_messages.append(HumanMessage(content=question))
        
        # Run RAG agent asynchronously
        try:
            rag_result = await rag_agent.ainvoke({"messages": rag_messages})
            rag_answer = rag_result["messages"][-1].content
            
            # Extract tools used from RAG messages
            tools_used = []
            for msg in rag_result["messages"]:
                if isinstance(msg, ToolMessage):
                    tools_used.append(msg.name)
            
            # Generate concise summary for voice
            summary = await generate_summary(rag_answer, max_words=SUMMARY_MAX_WORDS)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Store interaction to blob storage (async, non-blocking)
            # Only store if session_id is provided and category is not invalid
            if session_id and category != "invalid":
                asyncio.create_task(
                    store_interaction(
                        session_id=session_id,
                        question=question,
                        intent_result=intent_result,
                        full_response=rag_answer,
                        summary=summary,
                        summary_length=SUMMARY_MAX_WORDS,
                        tools_used=tools_used if tools_used else None,
                        response_time_ms=response_time_ms,
                    )
                )
            
            # Return summary for voice (full response stored in blob)
            return {
                "intent": intent_result.get("intent"),
                "category": "query",
                "module": intent_result.get("module"),
                "use_case": intent_result.get("use_case"),
                "answer": summary,
                "confidence": confidence,
                "requires_context": [],
                "entities": intent_result.get("entities", {})
            }
        except Exception as e:
            print(f"RAG error: {e}")
            # Fallback: return intent classification with error message
            return {
                "intent": intent_result.get("intent"),
                "category": "query",
                "module": intent_result.get("module"),
                "use_case": intent_result.get("use_case"),
                "answer": "I encountered an error retrieving information. Please try again.",
                "confidence": confidence * 0.5,  # Reduce confidence due to error
                "requires_context": [],
                "entities": intent_result.get("entities", {})
            }
    
    # Fallback (should not reach here)
    return {
        "intent": intent_result.get("intent", "invalid"),
        "category": "invalid",
        "module": None,
        "use_case": None,
        "answer": "I encountered an error processing your request. Please try again.",
        "confidence": 0.0,
        "requires_context": [],
        "entities": {}
    }