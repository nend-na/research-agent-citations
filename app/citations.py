"""
citations.py

Builds structured citation metadata for a set of retrieved chunks so the
UI/API can render "Source N -> filename, chunk #, similarity score" cards,
and can cross-check that the generated answer actually references sources
that were retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.retriever import RetrievedChunk


@dataclass
class Citation:
    index: int  # 1-based, matches [Source N] markers
    filename: str
    chunk_id: int
    similarity: float
    preview: str


def build_citations(chunks: List[RetrievedChunk], preview_chars: int = 220) -> List[Citation]:
    citations = []
    for i, rc in enumerate(chunks, start=1):
        preview = rc.chunk.text[:preview_chars].strip()
        if len(rc.chunk.text) > preview_chars:
            preview += "..."
        citations.append(
            Citation(
                index=i,
                filename=rc.chunk.doc_filename,
                chunk_id=rc.chunk.chunk_id,
                similarity=round(rc.score, 4),
                preview=preview,
            )
        )
    return citations


def referenced_source_indices(answer: str) -> List[int]:
    """Extract the set of [Source N] indices actually cited in an answer."""
    return sorted({int(m) for m in re.findall(r"\[Source (\d+)\]", answer)})


def unused_citations(answer: str, citations: List[Citation]) -> List[Citation]:
    """Citations that were retrieved but never referenced in the final answer."""
    used = set(referenced_source_indices(answer))
    return [c for c in citations if c.index not in used]
