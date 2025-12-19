"""
Chat Controller
Handles HTTP requests for the chat API endpoints.
"""
from fastapi import APIRouter, HTTPException
from app.models.request_models import Query, IntentResponse
from app.services.rag_service import process_with_intent_classification
from app.services.session_service import (
    get_session_history,
    add_to_session,
    clear_session,
    clear_all_sessions
)
from app.services.blob_storage_service import get_session_interactions

router = APIRouter()


@router.post("/ask", response_model=IntentResponse)
async def ask_question(query: Query):
    """
    Main endpoint for asking questions to the HR Voice Assistant.
    Processes questions with intent classification and RAG.
    """
    try:
        # Get conversation history for this session
        conversation_history = get_session_history(query.session_id)
        
        # Process question with intent classification and RAG
        result = await process_with_intent_classification(
            question=query.question,
            conversation_history=conversation_history if conversation_history else None,
            session_id=query.session_id
        )
        
        # Update session history if session_id is provided
        if query.session_id and result.get("answer"):
            add_to_session(
                session_id=query.session_id,
                user_message=query.question,
                ai_message=result.get("answer", "")
            )
        
        return IntentResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@router.post("/reset")
async def reset_session(session_id: str):
    """
    Reset conversation history for a session.
    """
    try:
        clear_session(session_id)
        return {"message": f"Session {session_id} reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting session: {str(e)}")


@router.get("/audit/{session_id}")
async def get_audit_trail(session_id: str):
    """
    Retrieve audit trail (stored interactions) for a session from blob storage.
    Returns both full_response and summary for each interaction.
    
    Note: Returns None if blob storage is not configured or session not found.
    """
    try:
        session_data = await get_session_interactions(session_id)
        if session_data is None:
            return {
                "session_id": session_id,
                "message": "No audit trail found. This could mean: blob storage not configured, session doesn't exist, or no interactions stored yet.",
                "data": None
            }
        return session_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving audit trail: {str(e)}")