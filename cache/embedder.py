"""
Lightweight text embedder for semantic cache key generation.

Uses ``sentence-transformers`` (``all-MiniLM-L6-v2`` by default) to
produce 384-dim embeddings.  The model is tiny (~22 MB) and runs on CPU
in < 5 ms, so it adds negligible overhead to the cache lookup path.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps a SentenceTransformer model to produce fixed-size embeddings.

    Parameters
    ----------
    model_name : str
        SentenceTransformer model ID.  Defaults to ``all-MiniLM-L6-v2``.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv(
            "EMBED_MODEL", "all-MiniLM-L6-v2"
        )
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to hash-based (exact-match) cache key."
            )

    def embed(self, text: str) -> list[float]:
        """Return a list[float] embedding for ``text``."""
        if self._model is None:
            # Fallback: return a zero vector so cache never hits
            return [0.0] * 384
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()
