"""
Blob Storage Service
Handles storage of interaction audit trails in Azure Blob Storage.
Stores session interactions as JSON files (one file per session).
"""
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from app.config.settings import (
    ENV,
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_CONTAINER_NAME,
)

# Try to import Azure Blob Storage SDK
try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import AzureError
    AZURE_BLOB_AVAILABLE = True
except ImportError:
    AZURE_BLOB_AVAILABLE = False
    print("[Blob Storage] Azure Blob Storage SDK not installed. Storage will be disabled.")


def _is_blob_storage_configured() -> bool:
    """Check if blob storage is configured and available."""
    if not AZURE_BLOB_AVAILABLE:
        return False
    
    # Check if we have at least one auth method configured
    has_connection_string = bool(AZURE_STORAGE_CONNECTION_STRING)
    has_account_auth = bool(AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY)
    
    return has_connection_string or has_account_auth


def _get_blob_service_client():
    """Get Azure Blob Service client."""
    if AZURE_STORAGE_CONNECTION_STRING:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    elif AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
        account_url = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
        return BlobServiceClient(account_url=account_url, credential=AZURE_STORAGE_ACCOUNT_KEY)
    else:
        raise ValueError("Azure Blob Storage credentials not configured")


async def store_interaction(
    session_id: str,
    question: str,
    intent_result: Dict[str, Any],
    full_response: str,
    summary: str,
    summary_length: int,
    tools_used: Optional[List[str]] = None,
    response_time_ms: Optional[float] = None,
) -> Optional[str]:
    """
    Store an interaction to Azure Blob Storage (one JSON file per session).
    Appends to existing session file or creates new one.
    
    Args:
        session_id: Session identifier
        question: User's question
        intent_result: Intent classification result
        full_response: Full RAG response text
        summary: Concise summary of the response
        summary_length: Maximum word count used for summary
        tools_used: List of tool names used (optional)
        response_time_ms: Response time in milliseconds (optional)
    
    Returns:
        Blob URL if successful, None if storage is not configured or fails
    """
    if not _is_blob_storage_configured():
        # Silently skip if not configured (common in dev)
        return None
    
    try:
        # Run in thread pool to avoid blocking (blob operations are sync)
        return await asyncio.to_thread(
            _store_interaction_sync,
            session_id,
            question,
            intent_result,
            full_response,
            summary,
            summary_length,
            tools_used,
            response_time_ms,
        )
    except Exception as e:
        # Log but don't fail the request
        print(f"[Blob Storage] Error storing interaction: {e}")
        return None


def _store_interaction_sync(
    session_id: str,
    question: str,
    intent_result: Dict[str, Any],
    full_response: str,
    summary: str,
    summary_length: int,
    tools_used: Optional[List[str]],
    response_time_ms: Optional[float],
) -> Optional[str]:
    """Synchronous version of store_interaction for thread pool execution."""
    try:
        blob_service_client = _get_blob_service_client()
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        
        # Ensure container exists
        try:
            container_client.create_container()
        except Exception:
            # Container likely already exists, ignore
            pass
        
        # Blob name: {session_id}.json
        blob_name = f"{session_id}.json"
        blob_client = container_client.get_blob_client(blob_name)
        
        # Read existing session data if it exists
        session_data = {
            "session_id": session_id,
            "created_at": None,
            "interactions": []
        }
        
        try:
            existing_blob = blob_client.download_blob()
            existing_content = existing_blob.readall().decode('utf-8')
            session_data = json.loads(existing_content)
        except Exception:
            # Blob doesn't exist yet, use default structure
            session_data["created_at"] = datetime.now(timezone.utc).isoformat()
        
        # Create new interaction entry
        interaction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "intent": intent_result,
            "full_response": full_response,
            "summary": summary,
            "summary_length": summary_length,
            "tools_used": tools_used or [],
            "metadata": {
                "env": ENV,
                "response_time_ms": response_time_ms,
            }
        }
        
        # Append interaction
        session_data["interactions"].append(interaction)
        
        # Write updated session data back to blob
        updated_content = json.dumps(session_data, indent=2, ensure_ascii=False)
        blob_client.upload_blob(
            updated_content,
            overwrite=True,
            content_settings={"content_type": "application/json"}
        )
        
        # Return blob URL
        blob_url = blob_client.url
        return blob_url
        
    except Exception as e:
        print(f"[Blob Storage] Error in _store_interaction_sync: {e}")
        return None


async def get_session_interactions(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve all interactions for a session from blob storage.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Session data dictionary or None if not found/configured
    """
    if not _is_blob_storage_configured():
        return None
    
    try:
        return await asyncio.to_thread(_get_session_interactions_sync, session_id)
    except Exception as e:
        print(f"[Blob Storage] Error retrieving session interactions: {e}")
        return None


def _get_session_interactions_sync(session_id: str) -> Optional[Dict[str, Any]]:
    """Synchronous version of get_session_interactions."""
    try:
        blob_service_client = _get_blob_service_client()
        container_client = blob_service_client.get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        blob_name = f"{session_id}.json"
        blob_client = container_client.get_blob_client(blob_name)
        
        existing_blob = blob_client.download_blob()
        content = existing_blob.readall().decode('utf-8')
        return json.loads(content)
        
    except Exception:
        # Blob doesn't exist or error reading
        return None

