"""
Basic tests for the retrieval + citation pipeline.
Run with: pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.chunking import chunk_documents, chunk_text
from app.citations import build_citations, referenced_source_indices, unused_citations
from app.embeddings import EmbeddingModel
from app.generator import NO_EVIDENCE_MESSAGE, generate_answer
from app.loader import RawDocument, load_documents_from_folder
from app.retriever import Retriever

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_docs")


def test_chunk_text_respects_overlap():
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 3
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    # Last 50 words of chunk 1 should reappear at the start of chunk 2.
    assert first_words[-50:] == second_words[:50]


def test_load_sample_documents():
    docs = load_documents_from_folder(SAMPLE_DIR)
    assert len(docs) == 3
    filenames = {d.filename for d in docs}
    assert "employee_report.txt" in filenames


def test_retrieval_returns_relevant_chunk():
    docs = load_documents_from_folder(SAMPLE_DIR)
    chunks = chunk_documents(docs)
    model = EmbeddingModel()
    retriever = Retriever(model)
    retriever.build(chunks)

    results = retriever.search("What causes employee attrition?", top_k=3)
    assert len(results) > 0
    assert any("employee_report" in r.chunk.doc_filename for r in results)


def test_missing_information_path():
    docs = [RawDocument(filename="unrelated.txt", text="Bananas are a good source of potassium.")]
    chunks = chunk_documents(docs)
    model = EmbeddingModel()
    retriever = Retriever(model)
    retriever.build(chunks)

    # Force the mock generator to demonstrate graceful handling when nothing fits.
    os_environ_backup = os.environ.get("LLM_PROVIDER")
    os.environ["LLM_PROVIDER"] = "mock"
    result = generate_answer("What are the major causes of employee attrition?", [])
    assert result.answer == NO_EVIDENCE_MESSAGE
    assert result.success is True
    if os_environ_backup is not None:
        os.environ["LLM_PROVIDER"] = os_environ_backup


def test_missing_api_key_reports_error_not_silent_success():
    docs = load_documents_from_folder(SAMPLE_DIR)
    chunks = chunk_documents(docs)
    model = EmbeddingModel()
    retriever = Retriever(model)
    retriever.build(chunks)
    results = retriever.search("What causes employee attrition?", top_k=3)

    backup = os.environ.get("LLM_PROVIDER")
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ["LLM_PROVIDER"] = "anthropic"

    result = generate_answer("What causes employee attrition?", results)
    assert result.success is False
    assert result.used_fallback is True
    assert result.error_type == "missing_api_key"
    assert result.answer  # mock fallback still produces something

    if backup is not None:
        os.environ["LLM_PROVIDER"] = backup
    else:
        os.environ.pop("LLM_PROVIDER", None)


def test_citation_building_and_unused_detection():
    docs = load_documents_from_folder(SAMPLE_DIR)
    chunks = chunk_documents(docs)
    model = EmbeddingModel()
    retriever = Retriever(model)
    retriever.build(chunks)

    results = retriever.search("What causes employee attrition?", top_k=3)
    citations = build_citations(results)
    assert len(citations) == len(results)

    fake_answer = "Compensation matters a lot [Source 1]."
    referenced = referenced_source_indices(fake_answer)
    assert referenced == [1]

    unused = unused_citations(fake_answer, citations)
    assert len(unused) == len(citations) - 1
