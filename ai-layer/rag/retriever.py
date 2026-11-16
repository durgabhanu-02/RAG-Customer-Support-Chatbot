from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_pdfs():
    pdfs = sorted([p for p in DOCS_DIR.rglob("*.pdf")])
    if not pdfs:
        raise SystemExit(f"No PDFs found. Put text PDFs into: {DOCS_DIR}")
    docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(str(pdf))
        loaded = loader.load()
        # Add filename metadata for citation
        for d in loaded:
            d.metadata["source_file"] = pdf.name
        docs.extend(loaded)
    return docs

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Load Documents
    print(f"📄 Loading PDFs from: {DOCS_DIR}")
    documents = load_pdfs()
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
    chunks = splitter.split_documents(documents)
    print(f"✂️  Split into {len(chunks)} chunks")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    print(f"🧠 Creating embeddings locally using: {EMBED_MODEL_NAME}")
    
    # Storing in FAISS vector store
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(INDEX_DIR))
    print(f"✅ FAISS index saved to: {INDEX_DIR}")

if __name__ == "__main__":
    main()

  
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
_db = None

def _load_db():
    global _db
    if _db is None:
        if not INDEX_DIR.exists():
            raise RuntimeError("RAG index not found. Run: python rag/build_index.py")
        _db = FAISS.load_local(
            str(INDEX_DIR),
            _embeddings,
            allow_dangerous_deserialization=True
        )
    return _db

def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Returns top-k chunks:
    [
      {"text": "...", "source_file": "abc.pdf", "page": 3},
      ...
    ]
    """
    db = _load_db()
    docs = db.similarity_search(query, k=k)
    results = []
    for d in docs:
        results.append({
            "text": d.page_content,
            "source_file": d.metadata.get("source_file", d.metadata.get("source", "unknown")),
            "page": d.metadata.get("page", None),
        })
    return results