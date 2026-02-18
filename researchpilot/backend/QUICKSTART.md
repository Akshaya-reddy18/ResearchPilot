# Quick Start Guide - ResearchPilot AI Agent

## 5-Minute Setup

### Step 1: Navigate to Backend
```bash
cd researchpilot/backend
```

### Step 2: Create & Activate Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Add Sample PDFs
Place PDF files in the `data/` directory (or skip for testing with empty vector store)

### Step 5: Start the Server
```bash
python main.py
```

✅ **Server running at**: `http://localhost:8000`

---

## Quick API Tests

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

### Test 2: Ingest Documents
```bash
curl -X POST http://localhost:8000/api/v1/papers/ingest
```

### Test 3: Search
```bash
curl -X GET "http://localhost:8000/api/v1/papers/search?query=research&top_k=5"
```

### Test 4: Interactive Documentation
Open in browser: **http://localhost:8000/api/docs**

---

## Directory Layout

```
researchpilot/backend/
├── main.py                  ← Run this to start server
├── requirements.txt         ← Dependencies
├── .env                     ← Configuration
├── README.md               ← Full documentation
├── QUICKSTART.md           ← This file
├── data/                   ← Add your PDFs here
├── vector_db/              ← Auto-created vector database
├── routers/
│   ├── papers.py          ← Search & ingest endpoints
│   └── chat.py            ← Chat & context endpoints
└── utils/
    ├── document_loader.py ← PDF processing
    ├── vector_store.py    ← Vector database
    └── research_assistant.py ← Context creation
```

---

## Key Features (Stage 1)

✅ FastAPI backend
✅ PDF document ingestion
✅ Vector-based semantic search
✅ ChromaDB integration
✅ Research context creation
✅ Interactive API documentation

---

## Environment Variables

Main configurations in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | 8000 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `DEBUG` | True | Debug mode |
| `DATA_DIR` | ./data | PDF directory |
| `VECTOR_DB_PATH` | ./vector_db | Database path |
| `MAX_CHUNK_SIZE` | 1000 | Text chunk size |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Embeddings model |

---

## Next Steps

1. ✅ Backend is running
2. 📚 Add PDF files to `data/` directory
3. 🔍 Call `/api/v1/papers/ingest` to process documents
4. 🔎 Use `/api/v1/papers/search` to find similar documents
5. 💬 Use `/api/v1/chat/chat` for context-aware queries

---

## Useful Links

- 📖 **Full Docs**: See `README.md`
- 🎯 **API Docs**: http://localhost:8000/api/docs
- 📑 **ReDoc**: http://localhost:8000/api/redoc

---

## Troubleshooting

**Issue: "Port 8000 already in use"**
```bash
# Change port in .env
PORT=8001
```

**Issue: Slow on first run**
- Normal! Downloading embeddings model (2-3 min)
- Subsequent runs are faster

**Issue: No PDFs found**
- Create `data/` folder
- Add `.pdf` files to it

---

**Ready to go!** 🚀
