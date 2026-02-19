"""
Papers Router Module
Handles document ingestion and semantic search endpoints.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

import os
from pathlib import Path
from utils.document_loader import DocumentLoader
from utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/papers", tags=["papers"])

# Initialize document loader and vector store
document_loader = None
vector_store = None


def initialize_papers_router(db_path: str, collection_name: str, data_dir: str):
    """
    Initialize the papers router with document loader and vector store.
    
    Args:
        db_path: Path to vector database
        collection_name: Name of the ChromaDB collection
        data_dir: Path to data directory with PDFs
    """
    global document_loader, vector_store
    # Use the db_path provided by main.py (single source of truth, already absolute)
    document_loader = DocumentLoader(data_dir=data_dir)
    vector_store = VectorStore(db_path=db_path, collection_name=collection_name)

    logger.info("Papers router initialized")


# Pydantic models for request/response
class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    """Individual search result model."""
    rank: int
    document: str
    similarity: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response model."""
    status: str
    query: str
    results_count: int
    results: List[SearchResult]


class IngestionResponse(BaseModel):
    """Ingestion response model."""
    status: str
    message: str
    documents_ingested: int


# Endpoints

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a PDF file to the data directory and ingest it into the vector store.
    
    This endpoint:
    1. Saves the uploaded PDF to the data directory
    2. Loads and chunks the PDF
    3. Generates embeddings
    4. Stores chunks in ChromaDB
    
    Args:
        file: PDF file to upload
        
    Returns:
        Dictionary with upload and ingestion status
    """
    try:
        if not document_loader or not vector_store:
            raise HTTPException(status_code=500, detail="Router not initialized")
        
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save file to data directory
        data_dir = Path(document_loader.data_dir)
        file_path = data_dir / file.filename
        
        # Handle duplicate filenames
        counter = 1
        original_name = file.filename
        while file_path.exists():
            name_parts = original_name.rsplit('.', 1)
            file_path = data_dir / f"{name_parts[0]}_{counter}.{name_parts[1]}"
            counter += 1
        
        # Write file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        
        # Load and process the uploaded PDF
        text = document_loader.load_pdf(str(file_path))
        metadata = {
            "source": file_path.name,
            "file_path": str(file_path),
            "document_type": "pdf"
        }
        chunks = document_loader.chunk_text(text, metadata)
        
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF")
        
        # Ingest into vector store
        result = vector_store.ingest_documents(chunks)
        
        if result["status"] == "success":
            logger.info(f"Successfully uploaded and ingested {file_path.name}: {result['count']} chunks")
            return {
                "status": "success",
                "message": f"File uploaded and ingested successfully",
                "filename": file_path.name,
                "documents_ingested": result["count"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_documents() -> Dict[str, Any]:
    """
    Ingest PDF documents from the data directory into the vector store.
    
    This endpoint:
    1. Loads all PDF files from the configured data directory
    2. Extracts and chunks text from PDFs
    3. Generates embeddings for each chunk
    4. Stores chunks in ChromaDB with metadata
    
    Returns:
        IngestionResponse with status and document count
        
    Raises:
        HTTPException: If ingestion fails
    """
    try:
        if not document_loader or not vector_store:
            raise HTTPException(status_code=500, detail="Router not initialized")
        
        logger.info("Starting document ingestion process...")
        
        # Load documents from directory
        documents = document_loader.load_documents_from_directory()
        
        if not documents:
            logger.warning("No documents found to ingest")
            return {
                "status": "warning",
                "message": "No PDF documents found in data directory. Please add PDFs to the data folder.",
                "documents_ingested": 0
            }
        
        # Ingest into vector store
        result = vector_store.ingest_documents(documents)
        
        if result["status"] == "success":
            logger.info(f"Successfully ingested {result['count']} document chunks")
            return {
                "status": "success",
                "message": result["message"],
                "documents_ingested": result["count"]
            }
        else:
            raise HTTPException(status_code=500, detail=result["message"])
    
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
async def search_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search for similar documents using semantic search.
    
    This endpoint:
    1. Takes a query string
    2. Generates embeddings for the query
    3. Searches the vector store using cosine similarity
    4. Returns top-k similar documents with scores
    
    Args:
        query: Search query string
        top_k: Number of top results to return (default: 5, max: 20)
        
    Returns:
        SearchResponse with query and matching documents
        
    Raises:
        HTTPException: If search fails or query is empty
    """
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        # Validate top_k
        top_k = min(max(1, top_k), 20)  # Ensure 1-20 range
        
        logger.info(f"Searching for query: '{query}' with top_k={top_k}")
        
        # Query vector store
        results = vector_store.query_similar_documents(query, top_k=top_k)
        
        # Format results
        formatted_results = [
            SearchResult(
                rank=result["rank"],
                document=result["document"],
                similarity=result["similarity"],
                metadata=result["metadata"]
            )
            for result in results
        ]
        
        logger.info(f"Found {len(formatted_results)} similar documents")
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/stats")
async def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the vector store collection.
    
    Returns:
        Dictionary with collection statistics including:
        - Collection name
        - Number of documents
        - Embedding dimension
        - Database path
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        stats = vector_store.get_collection_stats()
        return {
            "status": "success",
            "data": stats
        }
    
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/clear")
async def clear_collection() -> Dict[str, Any]:
    """
    Clear all documents from the vector store collection.
    
    WARNING: This action is irreversible.
    
    Returns:
        Dictionary with status of the clear operation
    """
    try:
        if not vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        success = vector_store.clear_collection()
        
        if success:
            logger.info("Collection cleared successfully")
            return {
                "status": "success",
                "message": "Collection cleared successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear collection")
    
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")
