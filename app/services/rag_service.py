"""
RAG Service
Business logic for RAG processing with intent classification routing.
"""
from typing import Dict, Any, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage
from app.services.intent_service import classify_intent
from app.agents.rag_agent import rag_agent


async def process_with_intent_classification(
    question: str, 
    conversation_history: Optional[List[BaseMessage]] = None
) -> Dict[str, Any]:
    """
    Process user question with intent classification and conditional RAG.
    Returns response matching schema.
    
    Args:
        question: User's input question
        conversation_history: Optional list of previous messages for context
    
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
    # Step 1: Intent Classification
    intent_result = classify_intent(question, conversation_history)
    
    category = intent_result.get("category", "").lower()
    confidence = intent_result.get("confidence", 0.0)
    
    # Step 2: Category-Based Routing
    
    # INVALID or low confidence: Return rejection
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
            
            return {
                "intent": intent_result.get("intent"),
                "category": "query",
                "module": intent_result.get("module"),
                "use_case": intent_result.get("use_case"),
                "answer": rag_answer,
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