"""
Chat Router Module
Exposes a simple chat endpoint wired to the ResearchAgent.
"""

import logging
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.research_agent import ResearchAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router (keep the same prefix to preserve API paths)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Agent will be initialized during app startup via initialize_chat_router
agent: Optional[ResearchAgent] = None


class ChatRequest(BaseModel):
    query: str
    use_context: bool = True
    top_k: int = 5


@router.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if agent is None:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    try:
        result = agent.analyze_topic(
            query=request.query,
            top_k=request.top_k,
            use_context=request.use_context,
        )
        return result
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def chat_health() -> Dict[str, str]:
    return {"status": "healthy", "service": "chat"}


def initialize_chat_router(db_path: str, collection_name: str, groq_api_key: str | None = None) -> None:
    """
    Initialize the chat router and the ResearchAgent instance.

    Args:
        db_path: Path to the Chroma vector DB directory.
        collection_name: Name of the Chroma collection to use.
        groq_api_key: Optional Groq API key; if provided, it will be set in the environment
    """
    global agent

    # If a Groq API key is provided via environment, ensure GroqClient can find it
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    # Create a ResearchAgent configured to use the specified DB/collection
    agent = ResearchAgent(db_path=db_path, collection_name=collection_name)

    logger.info("Chat router initialized with ResearchAgent")
