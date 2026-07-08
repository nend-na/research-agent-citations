"""
retriever.py

Builds a FAISS index over document chunk embeddings and retrieves the
top-k most similar chunks for a query using cosine similarity
(implemented via inner product on normalized vectors).

Falls back to a pure-numpy cosine similarity search if FAISS is not
installed, so the agent stays runnable in restricted environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from app.chunking import Chunk
from app.embeddings import EmbeddingModel

DEFAULT_TOP_K = 5

# Below this cosine similarity, a chunk is treated as "not actually relevant"
# rather than just the least-bad option in the corpus. Tuned for normalized
# MiniLM-style embeddings; the numpy/TF-IDF fallback uses the same scale
# since vectors are normalized in both paths.
DEFAULT_MIN_SCORE = 0.2


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.chunks: List[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._faiss_index = None
        self._use_faiss = False

        try:
            import faiss  # noqa: F401

            self._use_faiss = True
        except Exception as e:  # noqa: BLE001
            print(f"[retriever] Falling back to numpy search (faiss unavailable: {e})")

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        vectors = self.embedding_model.encode(texts)
        self._vectors = vectors

        if self._use_faiss and len(chunks) > 0:
            import faiss

            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on normalized vecs
            index.add(vectors)
            self._faiss_index = index

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> List[RetrievedChunk]:
        """Return up to top_k chunks with cosine similarity >= min_score.

        A plain top-k search always returns *something*, even if the corpus
        has nothing to do with the question (it just returns the least-bad
        match). Filtering by min_score lets the pipeline correctly report
        "not enough information" for out-of-domain questions instead of
        confidently citing irrelevant passages.
        """
        if not self.chunks:
            return []

        query_vec = self.embedding_model.encode([query])

        if self._use_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vec, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or float(score) < min_score:
                    continue
                results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
            return results

        # Numpy fallback: cosine similarity (vectors are already normalized).
        sims = self._vectors @ query_vec[0]
        top_indices = np.argsort(-sims)[:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(sims[i]))
            for i in top_indices
            if sims[i] >= min_score
        ]
