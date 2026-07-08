"""
main.py

FastAPI backend for the Research Agent with Citations.

Endpoints:
  POST /upload       - upload one or more documents (pdf/txt/md)
  POST /ask          - ask a question against the currently loaded documents
  GET  /documents     - list currently loaded documents
  DELETE /documents/{filename} - remove a document and rebuild the index
  GET  /health        - health check
"""

from __future__ import annotations

import os
import shutil
from typing import List

from dotenv import load_dotenv

load_dotenv()  # must run before any os.environ.get() calls below

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chunking import chunk_documents
from app.citations import build_citations, unused_citations
from app.embeddings import EmbeddingModel
from app.generator import generate_answer, check_provider_status
from app.loader import load_documents_from_folder, load_document
from app.retriever import Retriever, DEFAULT_TOP_K

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "data/uploads")
SAMPLE_DIR = os.environ.get("SAMPLE_DIR", "data/sample_docs")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
TOP_K = int(os.environ.get("TOP_K", str(DEFAULT_TOP_K)))

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Research Agent with Citations")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_embedding_model = EmbeddingModel()
_retriever = Retriever(_embedding_model)


def _rebuild_index() -> None:
    """Reload every document in UPLOAD_DIR + SAMPLE_DIR and rebuild the FAISS index."""
    docs = load_documents_from_folder(SAMPLE_DIR) + load_documents_from_folder(UPLOAD_DIR)
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    _retriever.build(chunks)


# Build the index once at startup so /ask works immediately with sample docs.
_rebuild_index()


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list
    unused_citations: list
    retrieval_notes: dict
    generation_status: dict


@app.get("/provider-status")
def provider_status():
    provider = os.environ.get("LLM_PROVIDER", "mock")
    return check_provider_status(provider)


@app.get("/config")
def config():
    return {
        "embedding_model": _embedding_model.model_name,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "similarity_metric": "cosine",
        "llm_provider": os.environ.get("LLM_PROVIDER", "mock"),
    }


class ProviderRequest(BaseModel):
    provider: str


@app.post("/set-provider")
def set_provider(req: ProviderRequest):
    provider = req.provider.lower().strip()
    if provider not in ("mock", "anthropic", "openai", "groq"):
        raise HTTPException(400, f"Unknown provider: {provider}")
    os.environ["LLM_PROVIDER"] = provider
    return check_provider_status(provider)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents_loaded": len({c.doc_filename for c in _retriever.chunks}),
    }


@app.get("/documents")
def list_documents():
    filenames = sorted({c.doc_filename for c in _retriever.chunks})
    return {"documents": filenames, "total_chunks": len(_retriever.chunks)}


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".pdf", ".txt", ".md"):
            raise HTTPException(400, f"Unsupported file type: {f.filename}")
        dest = os.path.join(UPLOAD_DIR, f.filename)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)

    _rebuild_index()
    return {"uploaded": saved, "total_chunks": len(_retriever.chunks)}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"Document not found: {filename}")
    os.remove(path)
    _rebuild_index()
    return {"deleted": filename, "total_chunks": len(_retriever.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    top_k = req.top_k or TOP_K
    retrieved = _retriever.search(req.question, top_k=top_k)

    if not retrieved:
        return AskResponse(
            question=req.question,
            answer="No relevant evidence was retrieved.",
            citations=[],
            unused_citations=[],
            retrieval_notes={
                "embedding_model": _embedding_model.model_name,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "top_k": top_k,
                "similarity_metric": "cosine",
            },
            generation_status={"success": True, "provider_used": "none", "error_type": None, "error_message": None},
        )

    result = generate_answer(req.question, retrieved)
    citations = build_citations(retrieved)
    unused = unused_citations(result.answer, citations)

    return AskResponse(
        question=req.question,
        answer=result.answer,
        citations=[c.__dict__ for c in citations],
        unused_citations=[c.__dict__ for c in unused],
        retrieval_notes={
            "embedding_model": _embedding_model.model_name,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": top_k,
            "similarity_metric": "cosine",
        },
        generation_status={
            "success": result.success,
            "requested_provider": result.requested_provider,
            "provider_used": result.provider_used,
            "used_fallback": result.used_fallback,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "model": result.model,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


# Mounted last so it never shadows the API routes above. Serves the
# premium static frontend (web/index.html, styles.css, app.js) at "/".
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
if os.path.isdir(_WEB_DIR):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_WEB_DIR, html=True), name="web")