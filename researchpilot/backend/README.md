# ResearchPilot AI Agent - Backend Setup & Usage Guide

## Project Overview

**ResearchPilot AI Agent** is an Autonomous Research Intelligence Hub built with FastAPI and vector-based semantic search. Stage 1 provides a foundation for document ingestion, vector storage, and research context creation.

### Tech Stack
- **Framework**: FastAPI with Uvicorn
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Document Processing**: PyPDF
- **Environment Management**: Python dotenv

## Project Structure

```
researchpilot/
└── backend/
    ├── main.py                      # FastAPI application entry point
    ├── requirements.txt             # Project dependencies
    ├── .env                         # Environment configuration
    ├── routers/
    │   ├── __init__.py
    │   ├── papers.py               # Document ingestion & search endpoints
    │   └── chat.py                 # Chat & context endpoints (placeholder)
    ├── utils/
    │   ├── __init__.py
    │   ├── document_loader.py      # PDF loading and text chunking
    │   ├── vector_store.py         # ChromaDB and embeddings
    │   └── research_assistant.py   # Research context creation
    ├── models/                      # Data models (future expansion)
    ├── data/                        # PDF documents for processing
    └── vector_db/                   # ChromaDB persistent storage
```

## Installation & Setup

### 1. Prerequisites
- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you encounter execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Navigate to backend directory
cd researchpilot/backend

# Install all required packages
pip install -r requirements.txt
```

**Expected installation output:**
```
Installing collected packages: fastapi, uvicorn, python-dotenv, sentence-transformers, 
chromadb, pypdf, pydantic, pydantic-settings, numpy, groq
Successfully installed...
```

### 4. Prepare Data Directory

```bash
# The data directory is created automatically
# Add your PDF files to: researchpilot/backend/data/

# Example structure:
# data/
#   ├── research_paper_1.pdf
#   ├── research_paper_2.pdf
#   └── research_paper_3.pdf
```

## Configuration

### Environment Variables (.env)

All configuration is managed through the `.env` file:

```env
# API Configuration
API_TITLE=ResearchPilot AI Agent
API_VERSION=1.0.0
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Vector Database
VECTOR_DB_PATH=./vector_db
CHROMA_COLLECTION_NAME=research_papers

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Document Processing
DATA_DIR=./data
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Groq API (placeholder for future)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=mixtral-8x7b-32768

# CORS
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]

# Logging
LOG_LEVEL=INFO
```

## Running the Server

### Start the Development Server

**Windows (PowerShell):**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Run the server
python main.py
```

**macOS/Linux:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Run the server
python main.py
```

**Expected output:**
```
2026-02-17 10:00:00,000 - __main__ - INFO - ============================================================
2026-02-17 10:00:00,000 - __main__ - INFO - ResearchPilot AI Agent - Startup
2026-02-17 10:00:00,000 - __main__ - INFO - ============================================================
2026-02-17 10:00:00,000 - __main__ - INFO - ✓ Papers router initialized
2026-02-17 10:00:00,000 - __main__ - INFO - ✓ Chat router initialized
2026-02-17 10:00:00,000 - __main__ - INFO - Server is ready to accept requests
2026-02-17 10:00:00,000 - utils.vector_store - INFO - ChromaDB client initialized...

INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Access the API

- **API Documentation (Swagger UI)**: http://localhost:8000/api/docs
- **ReDoc Documentation**: http://localhost:8000/api/redoc
- **Health Check**: http://localhost:8000/health

## API Endpoints

### Papers Router: Document Ingestion & Search

#### 1. **Ingest Documents**
```http
POST /api/v1/papers/ingest
```
Loads all PDF files from `/data` directory and stores embeddings.

**Response:**
```json
{
  "status": "success",
  "message": "Ingested 45 document chunks",
  "documents_ingested": 45
}
```

#### 2. **Search Documents**
```http
GET /api/v1/papers/search?query=neural networks&top_k=5
```
Semantic search using cosine similarity on embeddings.

**Response:**
```json
{
  "status": "success",
  "query": "neural networks",
  "results_count": 5,
  "results": [
    {
      "rank": 1,
      "document": "Neural networks are computing systems inspired by biological networks...",
      "similarity": 0.8734,
      "metadata": {
        "source": "paper1.pdf",
        "chunk_index": 2
      }
    }
  ]
}
```

#### 3. **Collection Statistics**
```http
GET /api/v1/papers/stats
```
Get information about the vector store collection.

**Response:**
```json
{
  "status": "success",
  "data": {
    "collection_name": "research_papers",
    "document_count": 45,
    "embedding_model": 384,
    "db_path": "./vector_db"
  }
}
```

#### 4. **Clear Collection**
```http
POST /api/v1/papers/clear
```
Remove all documents from the vector store (irreversible).

### Chat Router: Context & Intelligence

#### 1. **Chat with Context**
```http
POST /api/v1/chat/chat
Content-Type: application/json

{
  "query": "What are transformers in machine learning?",
  "use_context": true,
  "top_k": 5
}
```

**Response:**
```json
{
  "status": "success",
  "query": "What are transformers in machine learning?",
  "context_documents": 5,
  "context": "==================================================\nRESEARCH CONTEXT\n...",
  "message": "Retrieved 5 relevant documents for your query.",
  "note": "Full Groq integration pending..."
}
```

#### 2. **Create Research Context**
```http
POST /api/v1/chat/context
Content-Type: application/json

{
  "query": "explainability in AI",
  "top_k": 3
}
```

**Response:**
```json
{
  "status": "success",
  "query": "explainability in AI",
  "documents_retrieved": 3,
  "context": "research context with formatted papers..."
}
```

#### 3. **Chat Health**
```http
GET /api/v1/chat/health
```

## Usage Examples

### Using cURL

**Ingest Documents:**
```bash
curl -X POST http://localhost:8000/api/v1/papers/ingest
```

**Search:**
```bash
curl -X GET "http://localhost:8000/api/v1/papers/search?query=machine%20learning&top_k=3"
```

**Chat:**
```bash
curl -X POST http://localhost:8000/api/v1/chat/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain deep learning",
    "use_context": true,
    "top_k": 5
  }'
```

### Using Python Requests

```python
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

# Ingest documents
response = requests.post(f"{BASE_URL}/papers/ingest")
print(response.json())

# Search
response = requests.get(
    f"{BASE_URL}/papers/search",
    params={"query": "neural networks", "top_k": 5}
)
print(json.dumps(response.json(), indent=2))

# Chat with context
response = requests.post(
    f"{BASE_URL}/chat/chat",
    json={
        "query": "What is attention mechanism?",
        "use_context": True,
        "top_k": 5
    }
)
print(json.dumps(response.json(), indent=2))
```

### Using Swagger UI (Interactive)
1. Open http://localhost:8000/api/docs
2. Click on any endpoint
3. Click "Try it out"
4. Enter parameters and click "Execute"

## Architecture Components

### Document Loader (`utils/document_loader.py`)
- **Purpose**: Loading and chunking PDF files
- **Features**:
  - PDF text extraction
  - Intelligent text chunking (1000 chars, 200 overlap)
  - Metadata preservation
  - Batch processing

### Vector Store (`utils/vector_store.py`)
- **Purpose**: Embedding generation and vector database management
- **Features**:
  - ChromaDB integration
  - Sentence-Transformers embeddings
  - Semantic similarity search
  - Persistent storage
  - Collection statistics

### Research Assistant (`utils/research_assistant.py`)
- **Purpose**: Research context creation and future LLM integration
- **Current Features**:
  - Format retrieved documents
  - Create research context
  - Prepare for Groq integration (Stage 2)

### Papers Router (`routers/papers.py`)
- **Endpoints**: Document ingestion and semantic search
- **Features**:
  - `/ingest`: Load and process PDFs
  - `/search`: Semantic search with top-k results
  - `/stats`: Collection statistics
  - `/clear`: Clear vector store

### Chat Router (`routers/chat.py`)
- **Endpoints**: Chat and context creation
- **Features**:
  - `/chat`: Chat with document context
  - `/context`: Create research context
  - `/health`: Service health

## Future Enhancements (Stage 2+)

- [ ] Groq LLM integration for response generation
- [ ] Workspace management for multi-user support
- [ ] Authentication and authorization
- [ ] Advanced document metadata extraction
- [ ] Web frontend
- [ ] Document versioning
- [ ] Citation tracking
- [ ] Export capabilities (PDF, JSON, etc.)

## Troubleshooting

### Issue: "Module not found" error
**Solution**: Ensure virtual environment is activated and all dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Vector DB connection error
**Solution**: Ensure the `./vector_db` directory has write permissions
```bash
# Windows
icacls "vector_db" /grant:r %USERNAME%:F

# macOS/Linux
chmod -R 755 vector_db
```

### Issue: No PDFs found during ingestion
**Solution**: Ensure PDF files are in the `data/` directory
```bash
# Create sample data directory
mkdir data

# Add your PDF files to this directory
```

### Issue: Slow embeddings generation on first run
**Solution**: This is normal. The sentence-transformer model is being downloaded and cached. Subsequent runs will be faster.

## Performance Notes

- **First Load**: Model download and setup takes 2-3 minutes
- **Ingestion**: ~0.1-0.2 seconds per document chunk
- **Search**: ~0.05-0.1 seconds per query
- **Memory**: ~2-3 GB for embeddings model + vector store

## Logging

The application logs to console with configurable levels (DEBUG, INFO, WARNING, ERROR). Increase details:

```env
LOG_LEVEL=DEBUG
```

## Contributing Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints
- Add comments for complex logic
- Test endpoints before committing

## License

MIT License - ResearchPilot AI Agent

## Support

For issues, questions, or feature requests, please refer to the project documentation or contact the development team.

---

**Last Updated**: February 17, 2026
**Version**: 1.0.0 (Stage 1 - Foundation)
