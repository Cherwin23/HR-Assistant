"""
Request and Response Models for API endpoints.
"""
from pydantic import BaseModel
from typing import Dict, List, Optional


class Query(BaseModel):
    """Request model for /ask endpoint."""
    question: str  # required
    session_id: Optional[str] = None  # optional
    user_id: Optional[str] = None  # optional


class IntentResponse(BaseModel):
    """Response model matching Gateway Decision Engine Schema."""
    intent: str
    category: str
    module: Optional[str] = None
    use_case: Optional[str] = None
    answer: Optional[str] = None
    confidence: float
    requires_context: List[str] = []
    entities: Dict = {}