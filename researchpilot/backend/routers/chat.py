"""
Chat Router Module
Placeholder for AI chat endpoints with context from retrieved papers.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from utils.vector_store import VectorStore
from utils.research_assistant import ResearchAssistant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Initialize vector store and research assistant
vector_store = None
research_assistant = None


def initialize_chat_router(db_path: str, collection_name: str, groq_api_key: Optional[str] = None):
    """
    Initialize the chat router with vector store and research assistant.
    
    Args:
        db_path: Path to vector database
        collection_name: Name of the ChromaDB collection
        groq_api_key: Optional Groq API key for future implementation
    """
    global vector_store, research_assistant
    vector_store = VectorStore(db_path=db_path, collection_name=collection_name)
    research_assistant = ResearchAssistant(groq_api_key=groq_api_key)
    logger.info("Chat router initialized")


# Pydantic models for request/response

class ChatRequest(BaseModel):
    """Chat request model."""
    query: str
    use_context: bool = True
    top_k: int = 5


class ChatResponse(BaseModel):
    """Chat response model."""
    status: str
    query: str
    context_documents: int
    context: str
    message: str
    note: Optional[str] = None


class ContextRequest(BaseModel):
    """Request model for creating research context."""
    query: str
    top_k: int = 5


class ContextResponse(BaseModel):
    """Response model for research context."""
    status: str
    query: str
    documents_retrieved: int
    context: str


# Endpoints

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    Chat endpoint with context from retrieved papers.
    
    This is a placeholder endpoint that:
    1. Takes a user query
    2. Optionally retrieves relevant documents using semantic search
    3. Creates a research context from retrieved papers
    4. Returns the context (full Groq implementation pending)
    
    Args:
        request: ChatRequest containing query and parameters
        
    Returns:
        ChatResponse with query, context, and status
        
    Raises:
        HTTPException: If operation fails
        
    Note:
        Currently returns retrieved context only.
        Full LLM response generation will be added in Stage 2.
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not vector_store or not research_assistant:
            raise HTTPException(status_code=500, detail="Chat service not initialized")
        
        logger.info(f"Received chat request: {request.query}")
        
        # Initialize response
        context_documents = 0
        context = ""
        message = ""
        
        # Retrieve context if requested
        if request.use_context:
            logger.info(f"Retrieving context with top_k={request.top_k}")
            
            # Query vector store
            similar_docs = vector_store.query_similar_documents(
                request.query,
                top_k=request.top_k
            )
            context_documents = len(similar_docs)
            
            # Create research context
            context = research_assistant.create_research_context(
                similar_docs,
                request.query
            )
            
            if similar_docs:
                message = f"Retrieved {context_documents} relevant documents for your query."
            else:
                message = "No relevant documents found. The following are general insights based on your query."
        else:
            message = "Processing query without document context."
        
        # Placeholder note for future implementation
        note = "Full Groq integration pending. Currently returning retrieved context only."
        
        logger.info(f"Chat response prepared with {context_documents} documents")
        
        return {
            "status": "success",
            "query": request.query,
            "context_documents": context_documents,
            "context": context,
            "message": message,
            "note": note
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@router.post("/context", response_model=ContextResponse)
async def create_context(request: ContextRequest) -> Dict[str, Any]:
    """
    Create research context from retrieved documents.
    
    This endpoint:
    1. Takes a query
    2. Retrieves similar documents
    3. Formats them into a research context
    4. Returns the context for downstream processing
    
    Args:
        request: ContextRequest containing query and top_k
        
    Returns:
        ContextResponse with context and metadata
        
    Raises:
        HTTPException: If operation fails
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not vector_store or not research_assistant:
            raise HTTPException(status_code=500, detail="Service not initialized")
        
        logger.info(f"Creating context for query: {request.query}")
        
        # Query vector store
        similar_docs = vector_store.query_similar_documents(
            request.query,
            top_k=request.top_k
        )
        
        # Create context
        context = research_assistant.create_research_context(
            similar_docs,
            request.query
        )
        
        logger.info(f"Context created with {len(similar_docs)} documents")
        
        return {
            "status": "success",
            "query": request.query,
            "documents_retrieved": len(similar_docs),
            "context": context
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Context creation failed: {str(e)}")


@router.get("/health")
async def chat_health() -> Dict[str, str]:
    """
    Health check endpoint for chat service.
    
    Returns:
        Dictionary with service status
    """
    try:
        if vector_store and research_assistant:
            return {
                "status": "healthy",
                "service": "chat",
                "message": "Chat service is operational"
            }
        else:
            return {
                "status": "unhealthy",
                "service": "chat",
                "message": "Chat service components not initialized"
            }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "service": "chat",
            "message": f"Health check error: {str(e)}"
        }
