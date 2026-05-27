# AuditAI — RAG-Powered Policy Q&A System

An AI assistant that answers questions about your audit policy documents (COBIT, CISA, SOX frameworks) using **Retrieval-Augmented Generation (RAG)**. Built with LangChain, ChromaDB, FastAPI, and a custom dark-themed frontend.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | OpenAI `text-embedding-3-small` |
| Vector DB | ChromaDB (local, persistent) |
| LLM | GPT-4o-mini via LangChain |
| Retrieval | MMR (Maximal Marginal Relevance) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (no framework) |

---

## Project Structure

```
audit-rag/
├── backend/
│   ├── main.py          # FastAPI endpoints
│   ├── rag_chain.py     # LangChain RAG logic
│   └── ingest.py        # CLI ingestion script
├── frontend/
│   └── index.html       # Standalone UI (open in browser)
├── docs/                # Place your PDFs here
├── chroma_db/           # Auto-created after ingestion
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup (5 minutes)

### 1. Clone / download this project

```bash
cd audit-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and paste your key:
# OPENAI_API_KEY=sk-your-key-here
```

Get a key at https://platform.openai.com/api-keys  
A 50-page PDF costs roughly $0.01 to embed.

### 4. Add PDF documents to /docs

Recommended sources (free downloads):
- COBIT 2019 Framework Overview — ISACA website
- CISA Review Manual sample chapters
- Any SOX/ITGC control framework document
- Your own audit methodology documents

```bash
cp ~/Downloads/cobit-2019-framework.pdf docs/
```

### 5. Ingest documents

```bash
cd backend
python ingest.py
```

Output:
```
📄  Found 1 PDF(s):
    • cobit-2019-framework.pdf
⚙️   Chunking and embedding...
✅  Done! 312 chunks stored in ChromaDB.
```

### 6. Start the API server

```bash
uvicorn main:app --reload
```

Server running at: http://localhost:8000  
API docs at: http://localhost:8000/docs

### 7. Open the frontend

Simply open `frontend/index.html` in your browser. No build step needed.

---

## How It Works (RAG Pipeline)

```
User Question
    │
    ▼
OpenAI Embedding (query → vector)
    │
    ▼
ChromaDB MMR Search (top 5 relevant chunks)
    │
    ▼
LangChain ConversationalRetrievalChain
    │  (injects context + chat history into prompt)
    ▼
GPT-4o-mini generates grounded answer
    │
    ▼
FastAPI returns { answer, sources, session_id }
    │
    ▼
Frontend displays answer + source citations
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API + vectorstore status |
| POST | `/upload` | Upload a PDF file |
| POST | `/ingest` | Chunk, embed, and store all PDFs |
| POST | `/session/new` | Start a new conversation session |
| POST | `/chat` | Send a question, get answer + sources |
| GET | `/documents` | List uploaded PDFs |
| DELETE | `/session/{id}` | Clear a session |

Full interactive docs: http://localhost:8000/docs

---

## Example Questions

- "What are the five COBIT governance objectives?"
- "Explain IT General Controls and their three main domains"
- "What is the difference between ITGC and ITAC?"
- "How should access management controls be designed for SOX compliance?"
- "What control deficiencies qualify as material weaknesses?"

---

## Resume Description (copy-paste ready)

**AuditAI — RAG-Powered Policy Q&A System** | Python, LangChain, ChromaDB, FastAPI  
Designed and deployed an end-to-end Retrieval-Augmented Generation (RAG) system enabling semantic search over IT audit policy documents (COBIT, SOX, CISA frameworks). Implemented document chunking, OpenAI vector embeddings, MMR retrieval via ChromaDB, and a conversational LangChain chain with session memory. Built a FastAPI backend with REST endpoints and a custom dark-themed frontend UI. Deployed on [Render/Railway] with public demo link.

---

## Deployment (for portfolio link)

1. Push to GitHub
2. Create account on [Render.com](https://render.com) (free tier)
3. New Web Service → connect GitHub repo
4. Set environment variable: `OPENAI_API_KEY`
5. Build command: `pip install -r requirements.txt`
6. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

---

*Built as a portfolio project to demonstrate RAG pipeline development, LLM integration, and full-stack AI application design.*
