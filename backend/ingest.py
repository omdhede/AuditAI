"""
ingest.py
Run this script once (or whenever you add new PDFs) to chunk and embed documents.

Usage:
    cd backend
    python ingest.py

Place your PDF files in the /docs folder before running.
"""

import os
import sys
from pathlib import Path

# Add parent to path so we can import rag_chain
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌  OPENAI_API_KEY not found. Copy .env.example → .env and add your key.")
        sys.exit(1)

    docs_dir = Path(__file__).parent.parent / "docs"
    pdfs = list(docs_dir.glob("*.pdf"))

    if not pdfs:
        print(f"❌  No PDFs found in {docs_dir}")
        print("    Download COBIT 2019 Framework overview or any CISA study guide PDF")
        print("    and place it in the /docs folder.")
        sys.exit(1)

    print(f"📄  Found {len(pdfs)} PDF(s):")
    for p in pdfs:
        print(f"    • {p.name}")

    print("\n⚙️   Chunking and embedding... (this takes ~30 seconds for a 50-page doc)")

    from backend.rag_chain import ingest_documents
    chunks = ingest_documents(docs_dir)

    print(f"\n✅  Done! {chunks} chunks stored in ChromaDB.")
    print("    Start the API server: uvicorn main:app --reload")

if __name__ == "__main__":
    main()
