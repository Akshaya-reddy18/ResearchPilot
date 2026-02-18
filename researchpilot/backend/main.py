"""
ResearchPilot AI Agent - Main Application
FastAPI server for autonomous research intelligence hub.

This is the entry point for the ResearchPilot backend.
Features:
- Document ingestion with PDF processing
- Vector-based semantic search
- Research context creation
- Placeholder for future LLM integration

Stage 1: Foundation with vector database and document ingestion
Future stages: Groq integration, workspace management, authentication
"""

import logging
import os
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import routers
from routers import papers, chat

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan - startup and shutdown events.
    
    Startup:
    - Load environment variables
    - Initialize document loader and vector store
    - Initialize research assistant
    
    Shutdown:
    - Cleanup resources
    - Close database connections
    """
    # Startup
    logger.info("=" * 60)
    logger.info("ResearchPilot AI Agent - Startup")
    logger.info("=" * 60)
    
    try:
        # Get configuration from environment
        db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "research_papers")
        data_dir = os.getenv("DATA_DIR", "./data")
        groq_api_key = os.getenv("GROQ_API_KEY", None)
        
        # Initialize papers router
        papers.initialize_papers_router(db_path, collection_name, data_dir)
        logger.info("✓ Papers router initialized")
        
        # Initialize chat router
        chat.initialize_chat_router(db_path, collection_name, groq_api_key)
        logger.info("✓ Chat router initialized")
        
        logger.info("All services initialized successfully")
        logger.info("=" * 60)
        logger.info("Server is ready to accept requests")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield  # Server runs here
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("ResearchPilot AI Agent - Shutdown")
    logger.info("Cleaning up resources...")
    logger.info("=" * 60)


# Create FastAPI application
app = FastAPI(
    title=os.getenv("API_TITLE", "ResearchPilot AI Agent"),
    description="Autonomous Research Intelligence Hub with Vector Search",
    version=os.getenv("API_VERSION", "1.0.0"),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Configure CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]')
if isinstance(cors_origins, str):
    import json
    cors_origins = json.loads(cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS enabled for origins: {cors_origins}")

# Include routers
app.include_router(papers.router)
app.include_router(chat.router)

logger.info("Routers registered successfully")


# Root endpoint
@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing API information.
    
    Returns:
        Dictionary with API title, version, and available endpoints
    """
    return {
        "service": os.getenv("API_TITLE", "ResearchPilot AI Agent"),
        "version": os.getenv("API_VERSION", "1.0.0"),
        "status": "operational",
        "stage": "Stage 1 - Foundation & Vector Database",
        "endpoints": {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "papers_ingest": "POST /api/v1/papers/ingest",
            "papers_search": "GET /api/v1/papers/search?query=<query>",
            "papers_stats": "GET /api/v1/papers/stats",
            "chat": "POST /api/v1/chat/chat",
            "context": "POST /api/v1/chat/context",
            "chat_health": "GET /api/v1/chat/health"
        },
        "features": [
            "PDF document ingestion",
            "Vector-based semantic search",
            "Research context creation",
            "ChromaDB integration",
            "Sentence transformers embeddings"
        ],
        "environment": {
            "vector_db": os.getenv("VECTOR_DB_PATH", "./vector_db"),
            "data_directory": os.getenv("DATA_DIR", "./data"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        }
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Dictionary with service health status
    """
    return {
        "status": "healthy",
        "service": "ResearchPilot AI Agent",
        "message": "All systems operational"
    }


@app.get("/api/v1/status")
async def api_status() -> Dict[str, Any]:
    """
    Detailed API status endpoint.
    
    Returns:
        Dictionary with comprehensive service status information
    """
    return {
        "status": "operational",
        "version": os.getenv("API_VERSION", "1.0.0"),
        "stage": "Stage 1 - Foundation & Vector Database",
        "components": {
            "papers_router": "initialized",
            "chat_router": "initialized",
            "vector_store": "ready",
            "document_loader": "ready",
            "research_assistant": "ready"
        },
        "features": {
            "document_ingestion": True,
            "semantic_search": True,
            "context_creation": True,
            "groq_integration": False  # Placeholder for future
        },
        "configuration": {
            "vector_db_path": os.getenv("VECTOR_DB_PATH", "./vector_db"),
            "data_dir": os.getenv("DATA_DIR", "./data"),
            "collection_name": os.getenv("CHROMA_COLLECTION_NAME", "research_papers"),
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "chunk_size": int(os.getenv("MAX_CHUNK_SIZE", "1000")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200"))
        }
    }


if __name__ == "__main__":
    """
    Run the FastAPI application using Uvicorn.
    
    Configuration from environment variables:
    - HOST: Server host (default: 0.0.0.0)
    - PORT: Server port (default: 8000)
    - DEBUG: Debug mode (default: True)
    """
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "True").lower() == "true"
    
    logger.info(f"Starting Uvicorn server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
