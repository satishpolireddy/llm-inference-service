"""
Semantic cache backed by Redis.

On a cache hit the service skips LLM inference entirely, returning a
previously generated response for a semantically-similar prompt.  This
can reduce latency by 10× or more for repeated or near-duplicate queries.

Cache key
---------
A prompt embedding vector is computed via ``Embedder``.  At lookup time
we scan stored embeddings and return the stored text if the cosine
similarity exceeds ``similarity_threshold``.

Storage layout (Redis)
----------------------
``cache:emb:<sha256_hex>``  → JSON-serialised embedding list
``cache:txt:<sha256_hex>``  → Generated text string
``cache:keys``              → Redis Set of all stored SHA256 keys
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SemanticCache:
    """
    Redis-backed semantic cache for LLM completions.

    Parameters
    ----------
    redis_url : str
        Redis connection URL, e.g. ``"redis://localhost:6379"``.
    similarity_threshold : float
        Cosine similarity threshold for a cache hit (0–1). Default 0.92.
    ttl : int
        Time-to-live for cached entries in seconds. Default 3600 (1 hour).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        similarity_threshold: float = 0.92,
        ttl: int = 3600,
    ) -> None:
        self.redis_url = redis_url
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self._client = None
        self._embedder = None

    def _get_client(self):
        if self._client is None:
            import redis

            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _get_embedder(self):
        if self._embedder is None:
            from cache.embedder import Embedder

            self._embedder = Embedder()
        return self._embedder

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def get(self, prompt: str) -> Optional[str]:
        """
        Look up a prompt in the semantic cache.

        Returns the cached text on a hit, or ``None`` on a miss.
        """
        try:
            client = self._get_client()
            embedder = self._get_embedder()
            query_emb = embedder.embed(prompt)

            stored_keys = client.smembers("cache:keys")
            for sha in stored_keys:
                raw = client.get(f"cache:emb:{sha}")
                if raw is None:
                    continue
                stored_emb = json.loads(raw)
                sim = _cosine_similarity(query_emb, stored_emb)
                if sim >= self.similarity_threshold:
                    text = client.get(f"cache:txt:{sha}")
                    if text:
                        logger.debug("Cache hit (sim=%.3f) for prompt: %s…", sim, prompt[:60])
                        return text
        except Exception as exc:
            logger.warning("SemanticCache.get failed: %s", exc)
        return None

    async def set(self, prompt: str, text: str) -> None:
        """Store a prompt→text pair in the semantic cache."""
        try:
            client = self._get_client()
            embedder = self._get_embedder()
            emb = embedder.embed(prompt)
            sha = self._key(prompt)

            pipe = client.pipeline()
            pipe.set(f"cache:emb:{sha}", json.dumps(emb), ex=self.ttl)
            pipe.set(f"cache:txt:{sha}", text, ex=self.ttl)
            pipe.sadd("cache:keys", sha)
            pipe.execute()
            logger.debug("Cached response for prompt: %s…", prompt[:60])
        except Exception as exc:
            logger.warning("SemanticCache.set failed: %s", exc)

    def clear(self) -> int:
        """Flush all cache entries. Returns number of keys deleted."""
        try:
            client = self._get_client()
            keys = client.smembers("cache:keys")
            if not keys:
                return 0
            pipe = client.pipeline()
            for sha in keys:
                pipe.delete(f"cache:emb:{sha}")
                pipe.delete(f"cache:txt:{sha}")
            pipe.delete("cache:keys")
            pipe.execute()
            return len(keys)
        except Exception as exc:
            logger.warning("SemanticCache.clear failed: %s", exc)
            return 0
