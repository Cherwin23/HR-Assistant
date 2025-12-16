from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List, Optional
from app.rag import process_with_intent_classification
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

app = FastAPI(title="HR Voice Assistant API")

# MEMORY STORE (in-memory)
SESSION_MEMORY: Dict[str, List[BaseMessage]] = {}

# REQUEST BODY MODEL
class Query(BaseModel):
    question: str  # required
    session_id: Optional[str] = None  # optional
    user_id: Optional[str] = None  # optional

# RESPONSE MODEL
class IntentResponse(BaseModel):
    intent: str
    category: str
    module: Optional[str] = None
    use_case: Optional[str] = None
    answer: Optional[str] = None
    confidence: float
    requires_context: List[str] = []
    entities: Dict = {}

# MAIN CHAT ENDPOINT (with memory and intent classification)
@app.post("/ask", response_model=IntentResponse)
async def ask_question(payload: Query):
    """
    Gateway Decision Engine API /ask endpoint.
    Returns intent classification, category, answer (for queries), confidence, entities, and requires_context.
    """
    # Generate session_id if not provided
    session_id = payload.session_id or "default"
    
    # Create a new conversation if session_id doesn't exist yet
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    # Get conversation history for context
    conversation_history = SESSION_MEMORY[session_id]

    # Process with intent classification and conditional RAG (async)
    result = await process_with_intent_classification(
        question=payload.question,
        conversation_history=conversation_history if conversation_history else None
    )

    # Add user message to memory
    SESSION_MEMORY[session_id].append(HumanMessage(content=payload.question))
    
    # Add assistant response to memory (if answer exists)
    if result.get("answer"):
        SESSION_MEMORY[session_id].append(AIMessage(content=result["answer"]))

    # Return response
    return IntentResponse(**result)

# OPTIONAL: RESET SESSION MEMORY
@app.post("/reset")
async def reset_session(session_id: str):
    SESSION_MEMORY.pop(session_id, None)
    return {"status": f"Session '{session_id}' cleared."}