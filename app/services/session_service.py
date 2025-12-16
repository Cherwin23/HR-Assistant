"""
Session Service
Manages conversation history and session state.
"""
from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# In-memory session storage (replace with Redis/database in production)
_sessions: Dict[str, List[BaseMessage]] = {}


def get_session_history(session_id: Optional[str]) -> List[BaseMessage]:
    """
    Get conversation history for a session.
    
    Args:
        session_id: Optional session identifier
        
    Returns:
        List of messages in the conversation history
    """
    if not session_id:
        return []
    
    return _sessions.get(session_id, [])


def add_to_session(session_id: Optional[str], user_message: str, ai_message: str) -> None:
    """
    Add a user message and AI response to the session history.
    
    Args:
        session_id: Optional session identifier
        user_message: User's input message
        ai_message: AI's response message
    """
    if not session_id:
        return
    
    if session_id not in _sessions:
        _sessions[session_id] = []
    
    _sessions[session_id].append(HumanMessage(content=user_message))
    _sessions[session_id].append(AIMessage(content=ai_message))


def clear_session(session_id: Optional[str]) -> None:
    """
    Clear conversation history for a session.
    
    Args:
        session_id: Optional session identifier
    """
    if session_id and session_id in _sessions:
        _sessions[session_id] = []


def clear_all_sessions() -> None:
    """Clear all session histories."""
    _sessions.clear()