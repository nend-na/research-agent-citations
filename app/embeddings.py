"""
embeddings.py

Wraps a sentence-transformers embedding model (all-MiniLM-L6-v2 by default).
Falls back to a lightweight TF-IDF based embedding if sentence-transformers
or its model weights are not available, so the pipeline stays runnable
offline / without a large model download.
"""

from __future__ import annotations

from typing import List

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Thin wrapper that exposes a single `.encode(texts) -> np.ndarray` API."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._backend = "sentence-transformers"
        self._model = None
        self._tfidf = None

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception as e:  # noqa: BLE001
            print(
                f"[embeddings] Falling back to TF-IDF embeddings "
                f"(sentence-transformers unavailable: {e})"
            )
            self._backend = "tfidf"

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype="float32")

        if self._backend == "sentence-transformers":
            vectors = self._model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
            return vectors.astype("float32")

        return self._tfidf_encode(texts)

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=384, stop_words="english")
            self._tfidf.fit(texts)

        matrix = self._tfidf.transform(texts).toarray()
        # Pad/truncate to a fixed 384-dim space so it behaves like MiniLM output.
        if matrix.shape[1] < 384:
            pad = np.zeros((matrix.shape[0], 384 - matrix.shape[1]))
            matrix = np.hstack([matrix, pad])
        else:
            matrix = matrix[:, :384]

        matrix = normalize(matrix)
        return matrix.astype("float32")
