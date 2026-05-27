"""
main.py
FastAPI backend for the Audit Policy Q&A RAG system.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_chain import ingest_documents, build_chain, format_sources, DOCS_DIR, CHROMA_DIR

load_dotenv()

# ─── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AuditAI — RAG Q&A API",
    description="RAG-powered audit policy assistant using LangChain + ChromaDB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory session store (per conversation chain) ──────────────────────────
# In production, replace with Redis or a database
SESSION_CHAINS: dict = {}

# ─── Schema ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    question:   str

class IngestStatus(BaseModel):
    status:      str
    chunks_stored: Optional[int] = None
    message:     str


# ─── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "vectorstore_ready": CHROMA_DIR.exists()}


# ─── Upload PDFs ───────────────────────────────────────────────────────────────
@app.post("/upload", response_model=IngestStatus)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document to the docs/ folder.
    Accepts only PDF files.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = DOCS_DIR / file.filename
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return IngestStatus(
        status="uploaded",
        message=f"'{file.filename}' uploaded. Call /ingest to process it."
    )


# ─── Ingest documents ──────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestStatus)
def ingest():
    """
    Process all PDFs in docs/ — chunk, embed, and store in ChromaDB.
    Run this after uploading new documents.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in .env")

        chunks = ingest_documents()
        # Reset all sessions so they pick up fresh vectorstore
        SESSION_CHAINS.clear()

        return IngestStatus(
            status="success",
            chunks_stored=chunks,
            message=f"Ingestion complete. {chunks} chunks stored in ChromaDB."
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Start a new session ───────────────────────────────────────────────────────
@app.post("/session/new")
def new_session():
    """Create a new conversation session and return a session_id."""
    if not CHROMA_DIR.exists():
        raise HTTPException(
            status_code=400,
            detail="Vector store not found. Upload PDFs and call /ingest first."
        )
    session_id = str(uuid.uuid4())
    SESSION_CHAINS[session_id] = build_chain(session_id)
    return {"session_id": session_id}


# ─── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    """
    Send a question and receive a grounded answer with source citations.
    """
    if req.session_id not in SESSION_CHAINS:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /session/new first."
        )

    chain  = SESSION_CHAINS[req.session_id]

    try:
        result = chain({"question": req.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = format_sources(result.get("source_documents", []))

    return {
        "answer":   result["answer"],
        "sources":  sources,
        "session_id": req.session_id,
    }


# ─── List uploaded documents ──────────────────────────────────────────────────
@app.get("/documents")
def list_documents():
    """List all PDFs currently in the docs/ folder."""
    if not DOCS_DIR.exists():
        return {"documents": []}
    files = [f.name for f in DOCS_DIR.glob("*.pdf")]
    return {"documents": files, "count": len(files)}


# ─── Clear session ─────────────────────────────────────────────────────────────
@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    SESSION_CHAINS.pop(session_id, None)
    return {"status": "cleared"}


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
