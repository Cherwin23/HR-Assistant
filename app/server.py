from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from app.rag import rag_agent
from langchain_core.messages import HumanMessage, BaseMessage

app = FastAPI(title="HR Voice Assistant API")

# MEMORY STORE (in-memory)
SESSION_MEMORY: Dict[str, List[BaseMessage]] = {}

# REQUEST BODY MODEL
class Query(BaseModel):
    session_id: str
    question: str

# MAIN CHAT ENDPOINT (with memory)
@app.post("/ask")
async def ask_question(payload: Query):
    session_id = payload.session_id

    # Create a new conversation if session_id doesn't exist yet
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    # Add user message to memory
    SESSION_MEMORY[session_id].append(HumanMessage(content=payload.question))

    # Run the agent with the *entire* conversation history
    result = rag_agent.invoke({"messages": SESSION_MEMORY[session_id]})

    # Get the assistant's latest message
    answer_msg = result["messages"][-1]

    # Save assistant response to memory
    SESSION_MEMORY[session_id].append(answer_msg)

    # Return as API response
    return {
        "answer": answer_msg.content
    }

# OPTIONAL: RESET SESSION MEMORY
@app.post("/reset")
async def reset_session(session_id: str):
    SESSION_MEMORY.pop(session_id, None)
    return {"status": f"Session '{session_id}' cleared."}