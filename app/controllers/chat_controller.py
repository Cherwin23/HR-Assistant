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
            conversation_history=conversation_history if conversation_history else None
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