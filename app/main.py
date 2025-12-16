"""
Main FastAPI Application
Entry point for the HR Voice Assistant API.
"""
from fastapi import FastAPI
from app.controllers.chat_controller import router

app = FastAPI(
    title="HR Voice Assistant API",
    description="HR Voice Assistant with Intent Classification and RAG",
    version="1.0.0"
)

# Include routers
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "HR Voice Assistant API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}