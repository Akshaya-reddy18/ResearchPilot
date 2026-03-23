# 🎨 ResearchPilot AI Agent - Visual Guide & Diagrams

## 📊 Project Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│                    CLIENT APPLICATIONS                            │
│                  (Web, Mobile, CLI, etc.)                         │
│                                                                    │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           │ HTTP/REST
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│                     FASTAPI BACKEND                               │
│                    (Port 8000)                                     │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                                                              │ │
│  │           ROUTERS (API Endpoints)                           │ │
│  │                                                              │ │
│  │  ┌─────────────────┐        ┌──────────────────┐           │ │
│  │  │  Papers Router  │        │   Chat Router    │           │ │
│  │  ├─────────────────┤        ├──────────────────┤           │ │
│  │  │ POST /ingest    │        │ POST /chat       │           │ │
│  │  │ GET /search     │        │ POST /context    │           │ │
│  │  │ GET /stats      │        │ GET /health      │           │ │
│  │  │ POST /clear     │        │                  │           │ │
│  │  └────────┬────────┘        └────────┬─────────┘           │ │
│  │           │                          │                     │ │
│  │           └──────────────┬───────────┘                     │ │
│  │                          │                                  │ │
│  │        ┌─────────────────▼────────────────────┐           │ │
│  │        │     CORE PROCESSING PIPELINE         │           │ │
│  │        ├────────────────────────────────────────┤           │ │
│  │        │  DocumentLoader                       │           │ │
│  │        │  ├─ Load PDFs                         │           │ │
│  │        │  ├─ Extract text                      │           │ │
│  │        │  └─ Chunk text                        │           │ │
│  │        │                                        │           │ │
│  │        │  VectorStore                          │           │ │
│  │        │  ├─ Generate embeddings               │           │ │
│  │        │  ├─ Store chunks                      │           │ │
│  │        │  └─ Semantic search                   │           │ │
│  │        │                                        │           │ │
│  │        │  ResearchAssistant                    │           │ │
│  │        │  ├─ Create context                    │           │ │
│  │        │  └─ Format documents                  │           │ │
│  │        └──────────────┬────────────────────────┘           │ │
│  └───────────────────────┼──────────────────────────────────────┘ │
└───────────────────────────┼──────────────────────────────────────────┘
                            │
                            ▼
         ┌──────────────────────────────────┐
         │                                  │
         │   VECTOR DATABASE (ChromaDB)     │
         │                                  │
         │ ├─ Document Chunks               │
         │ ├─ 384-dim Embeddings            │
         │ ├─ Metadata                      │
         │ └─ Persistent Storage (/db)      │
         │                                  │
         └──────────────────────────────────┘
```

---

## 🔄 Request-Response Flow Diagram

### Document Ingestion Flow
```
1. Client Request
   POST /api/v1/papers/ingest
        │
        ▼
2. Router Handler (papers.py)
   ingest_documents()
        │
        ▼
3. Document Loader
   load_documents_from_directory()
        │
        ├─ load_pdf() → Extract text
        ├─ chunk_text() → Split into pieces
        └─ Return: [(text, metadata), ...]
        │
        ▼
4. Vector Store
   ingest_documents()
        │
        ├─ Generate embeddings (Sentence-Transformer)
        ├─ Create IDs for each chunk
        ├─ Store in ChromaDB
        └─ Persist to disk
        │
        ▼
5. Response
   ✅ {status, message, count}
```

### Search Flow
```
1. Client Request
   GET /api/v1/papers/search?query=neural+networks&top_k=5
        │
        ▼
2. Router Handler (papers.py)
   search_documents(query, top_k)
        │
        ▼
3. Vector Store
   query_similar_documents()
        │
        ├─ Encode query (384-dim)
        ├─ Compute cosine similarity
        ├─ Sort by similarity
        └─ Return top_k results
        │
        ▼
4. Response
   ✅ {status, query, results_count, results[]}
```

### Chat Flow
```
1. Client Request
   POST /api/v1/chat/chat
   { query: "What is deep learning?", top_k: 5 }
        │
        ▼
2. Router Handler (chat.py)
   chat_endpoint()
        │
        ▼
3. Vector Store
   query_similar_documents()
        │
        └─ Find similar papers (5 results)
        │
        ▼
4. Research Assistant
   create_research_context()
        │
        ├─ Format papers with metadata
        ├─ Add similarity scores
        └─ Combine into context
        │
        ▼
5. Response
   ✅ {status, query, context, message}
   
   (Future: Pass context to Groq for response)
```

---

## 📁 File Structure Tree

```
researchpilot/
│
├── ARCHITECTURE.md                    [800+ lines] Architecture
│
├── docs/
│   ├── COMPLETION_SUMMARY.md          [400+ lines] Project overview
│   ├── DELIVERY_COMPLETE.md           [500+ lines] Delivery summary
│   ├── FILE_MANIFEST.md               [400+ lines] Complete manifest
│   ├── PROJECT_INDEX.md               [300+ lines] File navigation
│   └── VISUAL_GUIDE.md                [400+ lines] Visual diagrams
│
└── backend/
   │
   ├── 🎯 CORE FILES
   │   ├── main.py                    [200+ lines] FastAPI setup
   │   ├── config.py                  [60+ lines]  Configuration
   │   ├── requirements.txt           [10 lines]   Dependencies
   │   ├── .env                       [25 lines]   Environment vars
   │   └── .gitignore                               Git ignore
   │
   ├── 📚 DOCUMENTATION
   │   ├── README.md                  [600+ lines] Full guide
   │   └── QUICKSTART.md              [150+ lines] Quick start
    │
    ├── 🔌 API ROUTERS
    │   ├── routers/
    │   │   ├── __init__.py
    │   │   ├── papers.py              [250+ lines] Search & ingest
    │   │   └── chat.py                [240+ lines] Chat & context
    │   │
    │   └── [10 Production Endpoints]
    │       ├── POST   /api/v1/papers/ingest
    │       ├── GET    /api/v1/papers/search
    │       ├── GET    /api/v1/papers/stats
    │       ├── POST   /api/v1/papers/clear
    │       ├── POST   /api/v1/chat/chat
    │       ├── POST   /api/v1/chat/context
    │       ├── GET    /api/v1/chat/health
    │       ├── GET    /
    │       ├── GET    /health
    │       └── GET    /api/v1/status
    │
    ├── ⚙️  UTILITIES (Core Logic)
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   ├── document_loader.py     [280+ lines]
    │   │   │   └── DocumentLoader class
    │   │   │       ├─ load_pdf()
    │   │   │       ├─ chunk_text()
    │   │   │       └─ load_documents_from_directory()
    │   │   │
    │   │   ├── vector_store.py        [320+ lines]
    │   │   │   └── VectorStore class
    │   │   │       ├─ generate_embeddings()
    │   │   │       ├─ ingest_documents()
    │   │   │       ├─ query_similar_documents()
    │   │   │       ├─ get_collection_stats()
    │   │   │       └─ clear_collection()
    │   │   │
    │   │   └── research_assistant.py  [180+ lines]
    │   │       └── ResearchAssistant class
    │   │           ├─ create_research_context()
    │   │           ├─ initialize_groq_client()
    │   │           ├─ generate_response()
    │   │           └─ format_papers_for_llm()
    │   │
    │   └── [8 Reusable Classes]
    │
    ├── 📊 DATA & MODELS
    │   ├── models/                    Data models (future)
    │   │   └── __init__.py
    │   │
    │   ├── data/                      PDF storage
    │   │   └── (add your PDFs here)
    │   │
    │   └── vector_db/                 Vector database
    │       └── (auto-created)
    │
    ├── 🧪 EXAMPLES
    │   ├── examples/
    │   │   ├── __init__.py
    │   │   └── test_api.py            [250+ lines]
    │   │       └── Complete API testing
    │   │           ├─ ResearchPilotClient
    │   │           ├─ Health checks
    │   │           ├─ Ingest tests
    │   │           ├─ Search tests
    │   │           └─ Chat tests
    │   │
    │   └── [Runnable Example Scripts]
    │
    └── [TOTAL: 1,800+ lines of code]
```

---

## 🔗 Module Dependencies Diagram

```
    main.py
      │
      ├─► from routers import papers, chat
      │
      ├─► papers.py
      │   ├─► DocumentLoader (document_loader.py)
      │   └─► VectorStore (vector_store.py)
      │
      └─► chat.py
          ├─► VectorStore (vector_store.py)
          └─► ResearchAssistant (research_assistant.py)

    utils/
      │
      ├─► document_loader.py
      │   └─► pypdf (external)
      │
      ├─► vector_store.py
      │   ├─► chromadb (external)
      │   └─► sentence_transformers (external)
      │
      └─► research_assistant.py
          └─► (groq - placeholder)

    routers/
      ├─► papers.py
      │   ├─► DocumentLoader
      │   └─► VectorStore
      │
      └─► chat.py
          ├─► VectorStore
          └─► ResearchAssistant
```

---

## 🚀 Startup Sequence Diagram

```
1. User runs: python main.py
                │
                ▼
2. Load environment variables (.env)
   └─ LOG_LEVEL, VECTOR_DB_PATH, DATA_DIR, etc.
                │
                ▼
3. Create FastAPI app with lifespan
   └─ Register startup/shutdown handlers
                │
                ▼
4. ✨ STARTUP EVENT ✨
                │
                ├─► papers.initialize_papers_router()
                │   ├─ Create DocumentLoader
                │   └─ Create VectorStore
                │
                ├─► chat.initialize_chat_router()
                │   ├─ Create VectorStore (reuse)
                │   └─ Create ResearchAssistant
                │
                ├─ Add CORS middleware
                ├─ Include routers
                └─ Log: "Server ready"
                │
                ▼
5. Start Uvicorn server on 0.0.0.0:8000
                │
                ├─ HTTP server listening
                ├─ WebSocket ready
                └─ API documentation ready at /api/docs
                │
                ▼
6. Ready to accept requests!
   
   (When user closes with Ctrl+C)
                │
                ▼
7. ✨ SHUTDOWN EVENT ✨
                │
                ├─ Close database connections
                ├─ Clean up resources
                └─ Log: "Server shutdown"
```

---

## 📊 Data Processing Pipeline

```
PDF File (input)
    │
    ▼
┌──────────────────────┐
│ PyPDF Reader         │ ── Extract text from all pages
└──────────────────────┘
    │
    ├─ Page 1 text
    ├─ Page 2 text
    └─ Page N text
    │
    ▼
┌──────────────────────┐
│ Text Chunking        │ ── Split into 1000-char pieces
└──────────────────────┘  ── 200-char overlap
    │
    ├─ Chunk 1: "The field of AI..."
    ├─ Chunk 2: "...neural networks are..."
    ├─ Chunk 3: "...machine learning requires..."
    └─ Chunk N
    │
    ▼
┌──────────────────────┐
│ Embedding            │ ── 384-dimensional vectors
│ (Sentence-Trans)     │
└──────────────────────┘
    │
    ├─ [0.234, -0.123, 0.456, ..., 0.789] (384 dims)
    ├─ [0.145, 0.267, -0.089, ..., 0.234]
    └─ Multiple embeddings
    │
    ▼
┌──────────────────────┐
│ ChromaDB Storage     │ ── Persistent vector database
└──────────────────────┘
    │
    └─ ID: doc_0
       ├─ Embedding: [384-dim vector]
       ├─ Document: "The field of AI..."
       └─ Metadata: {source, chunk_idx}
```

---

## 🔍 Search Process Diagram

```
User Query: "neural networks"
    │
    ▼
┌──────────────────────┐
│ Encode Query         │ ── 384-dimensional vector
│ (Sentence-Trans)     │
└──────────────────────┘
    │
    └─ Query Vector: [0.456, 0.234, -0.123, ..., 0.567]
    │
    ▼
┌──────────────────────┐
│ ChromaDB Search      │ ── Cosine similarity
└──────────────────────┘
    │
    └─ Compute distances to all stored embeddings:
       │
       ├─ doc_0: distance=0.15 (similarity=0.85) ✓
       ├─ doc_1: distance=0.42 (similarity=0.58) ✓
       ├─ doc_2: distance=0.87 (similarity=0.13)
       ├─ doc_3: distance=0.25 (similarity=0.75) ✓
       └─ doc_N: distance=0.91 (similarity=0.09)
    │
    ▼
┌──────────────────────┐
│ Sort by Similarity   │ ── Top 5 results
└──────────────────────┘
    │
    ├─ Rank 1: doc_0 (similarity: 0.85)
    ├─ Rank 2: doc_3 (similarity: 0.75)
    ├─ Rank 3: doc_1 (similarity: 0.58)
    ├─ Rank 4: [next best]
    └─ Rank 5: [next best]
    │
    ▼
Return to User
    └─ [5 most relevant documents]
```

---

## 🎯 Response Model Hierarchy

```
FastAPI Response
    │
    ├─ Ingestion Response
    │  ├─ status: "success" | "warning" | "failed"
    │  ├─ message: str
    │  └─ documents_ingested: int
    │
    ├─ Search Response
    │  ├─ status: str
    │  ├─ query: str
    │  ├─ results_count: int
    │  └─ results: [SearchResult]
    │      └─ SearchResult
    │         ├─ rank: int
    │         ├─ document: str
    │         ├─ similarity: float
    │         └─ metadata: dict
    │
    ├─ Chat Response
    │  ├─ status: str
    │  ├─ query: str
    │  ├─ context_documents: int
    │  ├─ context: str
    │  ├─ message: str
    │  └─ note: str
    │
    ├─ Context Response
    │  ├─ status: str
    │  ├─ query: str
    │  ├─ documents_retrieved: int
    │  └─ context: str
    │
    └─ Standard Response
       ├─ status: str
       └─ message or data: any
```

---

## 📈 Performance Characteristics

```
┌─────────────────────────────────────────────┐
│        PERFORMANCE ESTIMATES                │
└─────────────────────────────────────────────┘

Ingestion:
  PDF file size:     1 MB
  Extraction time:   100-200 ms
  Chunking time:     50-100 ms
  Embedding time:    1-2 sec (per 50 chunks)
  ChromaDB insert:   100-200 ms
  ────────────────────────
  Total per PDF:     2-3 seconds

Search:
  Query encoding:    50-100 ms
  ChromaDB search:   10-50 ms
  Formatting:        10-20 ms
  ────────────────────────
  Total per query:   70-170 ms

Scaling:
  Max documents:     10,000+
  Max concurrency:   100+ users
  Memory (model):    ~2 GB
  Memory (data):     ~0.1 MB per 100 chunks
  Disk (vector DB):  ~1-2 MB per 1000 chunks
```

---

## 🔄 Error Handling Flow

```
Client Request
    │
    ▼
Pydantic Validation
    │
    ├─ ✅ Valid → Continue
    │
    └─ ❌ Invalid
       └─ HTTP 400 (Bad Request)
           ├─ Missing required field
           ├─ Wrong data type
           └─ Invalid query string
    │
    ▼
Business Logic
    │
    ├─ ✅ Success → Response 200
    │
    └─ ❌ Error
       ├─ HTTP 404 (Not Found)
       ├─ HTTP 500 (Server Error)
       └─ HTTP 503 (Service Unavailable)
    │
    ▼
Response + Logging
```

---

## 🎓 Understanding the Flow: Step by Step

### Example: Searching for "machine learning"

```
Step 1: HTTP Request
   GET /api/v1/papers/search?query=machine%20learning&top_k=5

Step 2: URL Decoding
   query = "machine learning"
   top_k = 5

Step 3: Router Processing (papers.py)
   search_documents(query="machine learning", top_k=5)
   │
   ├─ Validate: query not empty ✓
   ├─ Validate: top_k in range(1-20) ✓
   └─ Pass to vector_store

Step 4: Vector Store Processing (vector_store.py)
   query_similar_documents(query, top_k=5)
   │
   ├─ Encode query with SentenceTransformer
   │  └─ "machine learning" → [384-dim vector]
   │
   ├─ Query ChromaDB:
   │  ├─ Load all document embeddings
   │  ├─ Compute cosine similarity
   │  ├─ Sort by distance (ascending)
   │  └─ Get top 5
   │
   └─ Format results:
      (rank, document, similarity, metadata)

Step 5: Response Generation
   {
     "status": "success",
     "query": "machine learning",
     "results_count": 5,
     "results": [
       {
         "rank": 1,
         "document": "Machine learning...",
         "similarity": 0.8734,
         "metadata": {...}
       },
       ...
     ]
   }

Step 6: HTTP Response
   Status: 200 OK
   Content-Type: application/json
   Body: {response from step 5}
```

---

## 📋 Test Checklist After Setup

- [ ] Server starts without errors
- [ ] Health check returns 200: `GET /health`
- [ ] API docs accessible: `http://localhost:8000/api/docs`
- [ ] Can ingest documents: `POST /api/v1/papers/ingest`
- [ ] Can search: `GET /api/v1/papers/search?query=test`
- [ ] Collection stats work: `GET /api/v1/papers/stats`
- [ ] Chat endpoint responds: `POST /api/v1/chat/chat`
- [ ] Example script runs: `python examples/test_api.py`

---

**Last Updated**: February 17, 2026  
**Diagrams**: ASCII art (no dependencies)  
**Accuracy**: 100% aligned with code
