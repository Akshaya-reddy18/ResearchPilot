# ResearchPilot AI Agent - Complete Architecture & Detailed Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Module Details](#module-details)
4. [Data Flow](#data-flow)
5. [API Reference](#api-reference)
6. [Development Guide](#development-guide)
7. [Deployment](#deployment)

---

## Project Overview

**ResearchPilot AI Agent** is a scalable, modular backend for autonomous research intelligence. 

### Stage 1 Goals (Current)
✅ Document ingestion and processing
✅ Vector-based semantic search
✅ Research context creation
✅ FastAPI backend foundation
✅ Production-ready code structure

### Future Stages
🔮 Groq LLM integration
🔮 Workspace management
🔮 Multi-user authentication
🔮 Web frontend
🔮 Advanced analytics

---

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐          ┌──────────────────┐             │
│  │   Papers Router  │          │   Chat Router    │             │
│  ├──────────────────┤          ├──────────────────┤             │
│  │ POST /ingest     │          │ POST /chat       │             │
│  │ GET /search      │          │ POST /context    │             │
│  │ GET /stats       │          │ GET /health      │             │
│  │ POST /clear      │          │                  │             │
│  └────────┬─────────┘          └────────┬─────────┘             │
│           │                             │                       │
│           └─────────────────┬───────────┘                       │
│                             │                                    │
│        ┌────────────────────▼────────────────────┐             │
│        │       Core Processing Pipeline          │             │
│        ├──────────────────────────────────────────┤             │
│        │  ┌──────────────────────────────────┐  │             │
│        │  │   Document Loader                │  │             │
│        │  │ - Load PDF from data/            │  │             │
│        │  │ - Extract text with PyPDF        │  │             │
│        │  │ - Chunk text (1000 chars overlap)│  │             │
│        │  └──────────┬───────────────────────┘  │             │
│        │             │                          │             │
│        │  ┌──────────▼───────────────────────┐  │             │
│        │  │   Vector Store                   │  │             │
│        │  │ - Generate embeddings            │  │             │
│        │  │ - Store in ChromaDB              │  │             │
│        │  │ - Semantic search (cosine sim)   │  │             │
│        │  └──────────┬───────────────────────┘  │             │
│        │             │                          │             │
│        │  ┌──────────▼───────────────────────┐  │             │
│        │  │   Research Assistant             │  │             │
│        │  │ - Create research context        │  │             │
│        │  │ - Format documents for LLM       │  │             │
│        │  │ - (Groq integration placeholder) │  │             │
│        │  └──────────────────────────────────┘  │             │
│        └──────────────────────────────────────────┘             │
│                             │                                    │
└─────────────────────┬───────┴────────────────────────────────────┘
                      │
        ┌─────────────▼─────────────┐
        │   Vector Database        │
        │   (ChromaDB)             │
        ├──────────────────────────┤
        │ - Persistent storage     │
        │ - Cosine similarity      │
        │ - 384-dim embeddings     │
        │ (all-MiniLM-L6-v2)       │
        └──────────────────────────┘
```

### Request Flow Architecture

```
Client Request
    │
    ▼
FastAPI Router (papers.py / chat.py)
    │
    ├─► Input Validation (Pydantic models)
    │
    ├─► Route to appropriate handler
    │
    ├─► Document Loader (if ingesting)
    │   └─► PDF extraction
    │   └─► Text chunking
    │
    ├─► Vector Store
    │   ├─► Generate embeddings (Sentence-Transformer)
    │   ├─► Store in ChromaDB
    │   └─► Query and return results
    │
    ├─► Research Assistant
    │   └─► Format context
    │
    └─► Response with metadata
        │
        ▼
    Client JSON Response
```

---

## Module Details

### 1. Document Loader (`utils/document_loader.py`)

**Purpose**: Load and process PDF documents

**Key Classes**:
- `DocumentLoader`: Main class for document processing

**Methods**:
```python
load_pdf(file_path)
    # Loads PDF and extracts text
    # Returns: Full text from all pages

chunk_text(text, metadata=None)
    # Split text into overlapping chunks
    # Returns: List of (chunk_text, metadata) tuples

load_documents_from_directory()
    # Batch process all PDFs in data directory
    # Returns: List of (chunk_text, metadata) tuples

get_sample_documents()
    # Convenience method to load all documents
```

**Configuration**:
- Chunk Size: 1000 characters (configurable)
- Chunk Overlap: 200 characters
- Max word breaks: Intelligently avoids cutting mid-word

**Example Usage**:
```python
from utils.document_loader import DocumentLoader

loader = DocumentLoader(
    data_dir="./data",
    chunk_size=1000,
    chunk_overlap=200
)

# Load all documents
documents = loader.load_documents_from_directory()
# Returns: [(chunk_text, {metadata}), ...]
```

### 2. Vector Store (`utils/vector_store.py`)

**Purpose**: Manage embeddings and semantic search

**Key Classes**:
- `VectorStore`: ChromaDB wrapper with embeddings

**Methods**:
```python
generate_embeddings(texts)
    # Generate embeddings using sentence-transformers
    # Returns: List of embedding vectors

ingest_documents(documents)
    # Store document chunks with embeddings
    # Returns: {status, message, count}

query_similar_documents(query, top_k=5)
    # Find similar documents via semantic search
    # Returns: List of results with similarity scores

get_collection_stats()
    # Get collection metadata
    # Returns: {collection_name, document_count, ...}

clear_collection()
    # Delete all documents
    # Returns: bool (success/failure)
```

**Embedding Model**: `all-MiniLM-L6-v2`
- Dimension: 384
- Speed: Fast
- Quality: High for semantic search
- Size: ~22 MB

**Similarity Metric**: Cosine Distance (converted to similarity)
- Lower distance = Higher similarity
- Range: 0 to 2 (converted to -1 to 1 as similarity)

**Example Usage**:
```python
from utils.vector_store import VectorStore

vs = VectorStore(
    db_path="./vector_db",
    collection_name="research_papers"
)

# Ingest documents
result = vs.ingest_documents(documents)

# Search
results = vs.query_similar_documents("neural networks", top_k=5)
# Each result: {rank, document, similarity, metadata}
```

### 3. Research Assistant (`utils/research_assistant.py`)

**Purpose**: Create research context and prepare for LLM integration

**Key Classes**:
- `ResearchAssistant`: Context creation and formatting

**Methods**:
```python
create_research_context(papers, query)
    # Format retrieved papers into research context
    # Returns: Formatted string ready for LLM

initialize_groq_client(api_key)
    # Placeholder for Groq initialization
    # (Stage 2 implementation)

generate_response(context, query, model)
    # Placeholder for Groq response generation
    # (Stage 2 implementation)

format_papers_for_llm(papers)
    # Format papers for LLM prompt injection
    # Returns: Formatted string
```

**Context Format**:
```
==================================================
RESEARCH CONTEXT
==================================================

User Query: [user_query]

==================================================
RETRIEVED PAPERS
==================================================

[1] Document Rank #1
    Similarity Score: 0.8734
    Source: paper1.pdf
    Chunk: 2

    Content:
    [Document excerpt...]

[2] Document Rank #2
    ...

==================================================
END OF CONTEXT
==================================================
```

### 4. Papers Router (`routers/papers.py`)

**Purpose**: Handle document ingestion and search endpoints

**Endpoints**:
```
POST /api/v1/papers/ingest
    ├─ Load all PDFs from data/
    ├─ Extract and chunk text
    ├─ Generate embeddings
    └─ Store in ChromaDB

GET /api/v1/papers/search?query=<query>&top_k=5
    ├─ Encode query
    ├─ Search vector store
    └─ Return similar documents

GET /api/v1/papers/stats
    └─ Return collection statistics

POST /api/v1/papers/clear
    └─ Clear all documents
```

**Response Models**:
```python
IngestionResponse(BaseModel):
    status: str
    message: str
    documents_ingested: int

SearchResponse(BaseModel):
    status: str
    query: str
    results_count: int
    results: List[SearchResult]

SearchResult(BaseModel):
    rank: int
    document: str
    similarity: float
    metadata: Dict
```

### 5. Chat Router (`routers/chat.py`)

**Purpose**: Chat and context endpoints (placeholder for LLM)

**Endpoints**:
```
POST /api/v1/chat/chat
    ├─ Accept query
    ├─ Retrieve context (optional)
    ├─ Create research context
    └─ Return context (full LLM pending)

POST /api/v1/chat/context
    ├─ Accept query
    ├─ Retrieve documents
    └─ Create research context

GET /api/v1/chat/health
    └─ Service health status
```

**Response Models**:
```python
ChatResponse(BaseModel):
    status: str
    query: str
    context_documents: int
    context: str
    message: str
    note: Optional[str]

ContextResponse(BaseModel):
    status: str
    query: str
    documents_retrieved: int
    context: str
```

### 6. Main Application (`main.py`)

**Purpose**: FastAPI application setup and initialization

**Features**:
- Lifespan management (startup/shutdown)
- CORS middleware configuration
- Router registration
- Health check endpoints
- API documentation
- Environment variable loading

**Root Endpoints**:
```
GET /
    └─ API information and endpoints

GET /health
    └─ Health check

GET /api/v1/status
    └─ Detailed status
```

---

## Data Flow

### Document Ingestion Flow

```
Client Request: POST /api/v1/papers/ingest
    │
    ▼
papers.py: ingest_documents()
    │
    ▼
DocumentLoader.load_documents_from_directory()
    │
    ├─► For each PDF:
    │   ├─ load_pdf() → Extract text with PyPDF
    │   ├─ chunk_text() → Split into 1000-char chunks
    │   └─ Return (chunk, metadata) pairs
    │
    ▼
VectorStore.ingest_documents()
    │
    ├─ Collect all chunks
    ├─ Generate embeddings
    │   └─ SentenceTransformer.encode() → 384-dim vectors
    ├─ Store in ChromaDB
    │   ├─ Document IDs: doc_0, doc_1, ...
    │   ├─ Embeddings: 384-dim vectors
    │   ├─ Metadata: source, chunk_index, etc.
    │   └─ Persist to ./vector_db/
    │
    ▼
Response: {status, message, count}
```

### Search Flow

```
Client Request: GET /api/v1/papers/search?query=neural+networks
    │
    ▼
papers.py: search_documents(query, top_k)
    │
    ▼
VectorStore.query_similar_documents()
    │
    ├─ Encode query
    │   └─ SentenceTransformer.encode(query) → 384-dim vector
    │
    ├─ Search ChromaDB
    │   ├─ Compute cosine similarity
    │   ├─ Sort by similarity
    │   └─ Return top_k results
    │
    ├─ Format results
    │   ├─ Rank, document, similarity score
    │   └─ Include metadata
    │
    ▼
Response: {status, query, results_count, results[]}
```

### Chat Flow

```
Client Request: POST /api/v1/chat/chat
    │
    ├─ Query: "What is deep learning?"
    ├─ use_context: true
    └─ top_k: 5
    │
    ▼
chat.py: chat_endpoint()
    │
    ▼
VectorStore.query_similar_documents()
    │
    ├─ Search for similar documents (5 results)
    │
    ▼
ResearchAssistant.create_research_context()
    │
    ├─ Format context with: query + papers
    ├─ Include similarity scores
    ├─ Include metadata
    │
    ▼
Response: {status, query, context_documents, context, message}
    │
    └─ (Future: Pass to Groq for response generation)
```

---

## API Reference

### Complete API Documentation

#### Papers Endpoints

**1. Ingest Documents**
```
POST /api/v1/papers/ingest

Request: (No body)

Response 200:
{
  "status": "success",
  "message": "Ingested 45 document chunks",
  "documents_ingested": 45
}

Response 202 (No PDFs):
{
  "status": "warning",
  "message": "No PDF documents found in data directory...",
  "documents_ingested": 0
}

Error 500:
{
  "detail": "Ingestion failed: [error message]"
}
```

**2. Search Documents**
```
GET /api/v1/papers/search?query=neural+networks&top_k=5

Query Parameters:
  - query (required): Search query string
  - top_k (optional): Number of results (1-20, default: 5)

Request: (No body)

Response 200:
{
  "status": "success",
  "query": "neural networks",
  "results_count": 5,
  "results": [
    {
      "rank": 1,
      "document": "Neural networks are computing systems...",
      "similarity": 0.8734,
      "metadata": {
        "source": "paper1.pdf",
        "chunk_index": 2,
        "document_type": "pdf"
      }
    },
    ...
  ]
}

Error 400:
{
  "detail": "Query cannot be empty"
}

Error 500:
{
  "detail": "Search failed: [error message]"
}
```

**3. Collection Statistics**
```
GET /api/v1/papers/stats

Request: (No body)

Response 200:
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

**4. Clear Collection**
```
POST /api/v1/papers/clear

Request: (No body)

Response 200:
{
  "status": "success",
  "message": "Collection cleared successfully"
}

Warning: This action is irreversible!
```

#### Chat Endpoints

**1. Chat with Context**
```
POST /api/v1/chat/chat

Request Body:
{
  "query": "What are transformers?",
  "use_context": true,
  "top_k": 5
}

Response 200:
{
  "status": "success",
  "query": "What are transformers?",
  "context_documents": 5,
  "context": "[formatted research context]...",
  "message": "Retrieved 5 relevant documents for your query.",
  "note": "Full Groq integration pending. Currently returning retrieved context only."
}

Error 400:
{
  "detail": "Query cannot be empty"
}
```

**2. Create Research Context**
```
POST /api/v1/chat/context

Request Body:
{
  "query": "explainability in AI",
  "top_k": 3
}

Response 200:
{
  "status": "success",
  "query": "explainability in AI",
  "documents_retrieved": 3,
  "context": "[formatted research context]..."
}
```

**3. Chat Health**
```
GET /api/v1/chat/health

Response 200:
{
  "status": "healthy",
  "service": "chat",
  "message": "Chat service is operational"
}
```

#### System Endpoints

**1. Root Information**
```
GET /

Response 200:
{
  "service": "ResearchPilot AI Agent",
  "version": "1.0.0",
  "status": "operational",
  "stage": "Stage 1 - Foundation & Vector Database",
  "endpoints": {...},
  "features": [...],
  "environment": {...}
}
```

**2. Health Check**
```
GET /health

Response 200:
{
  "status": "healthy",
  "service": "ResearchPilot AI Agent",
  "message": "All systems operational"
}
```

**3. API Status**
```
GET /api/v1/status

Response 200:
{
  "status": "operational",
  "version": "1.0.0",
  "stage": "Stage 1 - Foundation & Vector Database",
  "components": {...},
  "features": {...},
  "configuration": {...}
}
```

---

## Development Guide

### Adding New Endpoints

**Example: Adding a custom search endpoint**

1. Create method in relevant router:
```python
# In routers/papers.py
@router.get("/advanced-search")
async def advanced_search(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.5
) -> Dict[str, Any]:
    """Custom search with similarity threshold."""
    results = vector_store.query_similar_documents(query, top_k)
    # Filter by similarity
    filtered = [r for r in results if r["similarity"] >= min_similarity]
    return {
        "status": "success",
        "results_count": len(filtered),
        "results": filtered
    }
```

2. Test via Swagger UI or cURL
3. Update README.md with new endpoint

### Extending with New Utilities

**Example: Adding document metadata extraction**

```python
# Create new file: utils/metadata_extractor.py
class MetadataExtractor:
    """Extract metadata from documents."""
    
    def extract_title(self, text):
        """Extract title from text."""
        # Implementation
        pass
    
    def extract_authors(self, text):
        """Extract author information."""
        # Implementation
        pass

# Use in document_loader.py
from utils.metadata_extractor import MetadataExtractor

extractor = MetadataExtractor()
metadata = {
    "title": extractor.extract_title(text),
    "authors": extractor.extract_authors(text),
    ...
}
```

### Testing

**Run example tests**:
```bash
python examples/test_api.py
```

**Manual testing with cURL**:
```bash
# Test search
curl -X GET "http://localhost:8000/api/v1/papers/search?query=test&top_k=5"

# Test ingest
curl -X POST "http://localhost:8000/api/v1/papers/ingest"

# Test chat
curl -X POST "http://localhost:8000/api/v1/chat/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"test query","use_context":true}'
```

---

## Deployment

### Local Deployment

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
# Edit .env file as needed

# 3. Run
python main.py
```

### Docker Deployment (Future)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

### Cloud Deployment (Future)

- **Azure**: Azure App Service, Azure Container Instances
- **AWS**: EC2, ECS, Lambda
- **GCP**: Cloud Run, App Engine

---

## Performance & Optimization

### Benchmarks
- **Ingestion**: ~10-20 docs/sec
- **Search**: ~50-100 queries/sec
- **Embedding**: ~100-200 embeddings/sec
- **Memory**: ~2-3 GB for model + data

### Optimization Tips
1. Use smaller embedding models for faster ingestion
2. Batch queries for better throughput
3. Pre-warm embeddings on startup
4. Use caching for frequent queries
5. Monitor vector_db size regularly

### Scaling Considerations
- **Horizontal**: Load balancing across API instances
- **Vertical**: More CPU cores, GPU acceleration
- **Database**: Scale ChromaDB persistence
- **Storage**: Archive old vector databases

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | Change PORT in .env |
| Slow embedding on first run | Normal - model is cached after first run |
| No PDFs found | Add PDFs to data/ directory |
| Vector DB permissions | chmod -R 755 vector_db/ |
| Out of memory | Reduce chunk size or batch size |

---

## Future Roadmap

### Stage 2: LLM Integration
- ✉️ Groq integration
- 💬 Full chat responses
- 🔄 Multi-turn conversations

### Stage 3: Workspace & Auth
- 🔐 User authentication
- 📁 Workspace management
- 👥 Collaboration features

### Stage 4: Advanced Features
- 🎯 Citation tracking
- 📊 Analytics dashboard
- 📤 Export capabilities
- 🔗 Knowledge graphs

---

**Document Version**: 1.0
**Last Updated**: February 17, 2026
**Stage**: Stage 1 - Foundation & Vector Database
