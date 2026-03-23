"""
Configuration Module
Centralized configuration management for the application.

Future expansion: Add workspace-specific configs, user preferences, etc.
"""

import os
from typing import List
from pathlib import Path

class Config:
    """
    Application configuration class.
    
    Loads settings from environment variables with sensible defaults.
    """
    
    # API Configuration
    API_TITLE = os.getenv("API_TITLE", "ResearchPilot AI Agent")
    API_VERSION = os.getenv("API_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Database Configuration
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "research_papers")
    
    # Embeddings Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Document Processing Configuration
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Groq API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
    GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8080"]')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_cors_origins(cls) -> List[str]:
        """Parse CORS origins from environment variable."""
        import json
        if isinstance(cls.CORS_ORIGINS, str):
            try:
                return json.loads(cls.CORS_ORIGINS)
            except json.JSONDecodeError:
                return ["http://localhost:3000"]
        return cls.CORS_ORIGINS
    
    @classmethod
    def validate_paths(cls) -> bool:
        """Validate and create necessary directories."""
        try:
            Path(cls.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)
            Path(cls.DATA_DIR).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
