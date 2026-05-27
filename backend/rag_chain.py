"""
rag_chain.py
Core RAG logic: PDF ingestion, embedding, retrieval, and answer generation.
"""

import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


# ─── Config ────────────────────────────────────────────────────────────────────
DOCS_DIR     = Path(__file__).parent.parent / "docs"
CHROMA_DIR   = Path(__file__).parent.parent / "chroma_db"
EMBED_MODEL  = "text-embedding-3-small"   # cost-efficient, high quality
LLM_MODEL    = "gpt-4o-mini"              # fast + affordable for demos
CHUNK_SIZE   = 800
CHUNK_OVERLAP= 120
TOP_K        = 5                          # retrieved chunks per query


# ─── System prompt tailored for audit context ──────────────────────────────────
AUDIT_SYSTEM_PROMPT = """You are AuditAI, an expert assistant specialising in IT audit, 
SOX compliance, ITGC controls, COBIT framework, and CISA examination preparation.

Answer questions using ONLY the context retrieved from the uploaded documents.
If the context does not contain enough information, say so clearly — do not hallucinate.

Structure your answers clearly:
- Lead with a direct answer
- Support with specific references from the documents
- Add practical audit application notes where relevant

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

AUDIT_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=AUDIT_SYSTEM_PROMPT,
)


# ─── Ingest PDFs ───────────────────────────────────────────────────────────────
def ingest_documents(docs_dir: Path = DOCS_DIR) -> int:
    """
    Load all PDFs from docs_dir, chunk them, embed, and persist to ChromaDB.
    Returns the number of chunks stored.
    """
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    # Load all pages from all PDFs
    all_docs = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        # Tag each page with its source filename
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        all_docs.extend(pages)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    # Embed and persist
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.persist()
    return len(chunks)


# ─── Load existing vector store ────────────────────────────────────────────────
def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )


# ─── Build RAG chain ───────────────────────────────────────────────────────────
def build_chain(session_id: str) -> ConversationalRetrievalChain:
    """
    Build a ConversationalRetrievalChain with per-session memory.
    """
    vectorstore = load_vectorstore()
    retriever   = vectorstore.as_retriever(
        search_type="mmr",           # Maximal Marginal Relevance — diverse results
        search_kwargs={"k": TOP_K, "fetch_k": 15},
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, streaming=True)

    memory = ConversationBufferWindowMemory(
        k=6,                         # keep last 6 exchanges
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": AUDIT_PROMPT},
        return_source_documents=True,
        output_key="answer",
    )
    return chain


# ─── Source citation formatter ─────────────────────────────────────────────────
def format_sources(source_docs: list) -> List[dict]:
    """Extract unique source references for display in the UI."""
    seen   = set()
    sources = []
    for doc in source_docs:
        key = (doc.metadata.get("source_file", "Unknown"), doc.metadata.get("page", "?"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("source_file", "Unknown"),
                "page": doc.metadata.get("page", "?"),
                "snippet": doc.page_content[:180].strip() + "…",
            })
    return sources
