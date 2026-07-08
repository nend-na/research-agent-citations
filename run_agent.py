"""Command-line interface for the Research Agent with Citations."""
from __future__ import annotations

import os
import textwrap
from dotenv import load_dotenv
from app.chunking import chunk_documents
from app.citations import build_citations, unused_citations
from app.embeddings import EmbeddingModel
from app.generator import check_provider_status, generate_answer
from app.loader import load_documents_from_folder
from app.retriever import Retriever

SAMPLE_DIR = os.environ.get("SAMPLE_DIR", "data/sample_docs")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "data/uploads")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
TOP_K = int(os.environ.get("TOP_K", "5"))


def load_documents() -> list:
    docs = load_documents_from_folder(SAMPLE_DIR)
    if os.path.isdir(UPLOAD_DIR):
        docs += load_documents_from_folder(UPLOAD_DIR)
    return docs


def build_retriever() -> Retriever:
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No documents found in {SAMPLE_DIR} or {UPLOAD_DIR}.")

    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    model = EmbeddingModel()
    retriever = Retriever(model)
    retriever.build(chunks)
    return retriever


def summarize_status() -> str:
    provider = os.environ.get("LLM_PROVIDER", "mock")
    status = check_provider_status(provider)
    lines = [f"Provider: {provider}", f"Ready: {status.get('ready')}", f"Detail: {status.get('detail')}" ]
    if status.get("error_type"):
        lines.append(f"Error: {status.get('error_type')} - {status.get('detail')}")
    return "\n".join(lines)


def format_citations(citations: list) -> str:
    if not citations:
        return "No citations generated."
    rows = []
    for idx, c in enumerate(citations, start=1):
        rows.append(f"[{idx}] {c.doc_filename} (chunk {c.chunk_id}, score={c.score:.3f})")
        if preview := getattr(c, "preview", None):
            rows.append(f"    preview: {preview}")
    return "\n".join(rows)


def main() -> None:
    load_dotenv()
    print("Research Agent with Citations")
    print("============================\n")
    print(summarize_status())
    print("\nLoading documents and building the retrieval index...")

    retriever = build_retriever()
    print(f"Loaded documents from {SAMPLE_DIR} and {UPLOAD_DIR}.")
    print(f"Retrieved top_k={TOP_K}, chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}.\n")

    print("Type a research question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question or question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        retrieved = retriever.search(question, top_k=TOP_K)
        if not retrieved:
            print("No relevant evidence was retrieved. Try a different question or add documents.\n")
            continue

        result = generate_answer(question, retrieved)
        citations = build_citations(retrieved)
        unused = unused_citations(result.answer, citations)

        print("\nAnswer:\n")
        print(textwrap.fill(result.answer, width=96))
        print("\nCitations:\n")
        print(format_citations(citations))
        if unused:
            print("\nUnused citation sources detected:\n")
            print(format_citations(unused))

        print("\nGeneration status:")
        print(f"  provider_used: {result.provider_used}")
        print(f"  success: {result.success}")
        print(f"  used_fallback: {result.used_fallback}")
        print(f"  error_type: {result.error_type}")
        print(f"  error_message: {result.error_message}\n")


if __name__ == "__main__":
    main()
