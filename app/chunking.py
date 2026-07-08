"""
chunking.py

Splits document text into overlapping chunks suitable for embedding.
Chunk size and overlap are measured in whitespace-separated tokens
(a simple, dependency-free approximation of token count).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.loader import RawDocument

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    doc_filename: str
    chunk_id: int
    text: str


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += step

    return chunks


def chunk_documents(
    documents: List[RawDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(Chunk(doc_filename=doc.filename, chunk_id=i, text=piece))
    return all_chunks
