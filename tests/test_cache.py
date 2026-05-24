"""
Unit tests for the semantic cache layer.

Run with:
    pytest tests/test_cache.py -v
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from cache.redis_cache import SemanticCache, _cosine_similarity


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_normalized_vectors(self):
        # 45-degree angle → similarity = cos(π/4) ≈ 0.707
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1 / math.sqrt(2)
        assert _cosine_similarity(a, b) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# SemanticCache — Redis mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis():
    client = MagicMock()
    client.smembers.return_value = set()
    client.pipeline.return_value.__enter__ = lambda s: s
    client.pipeline.return_value.__exit__ = MagicMock(return_value=False)
    pipe = MagicMock()
    client.pipeline.return_value = pipe
    pipe.execute = MagicMock()
    pipe.set = MagicMock()
    pipe.sadd = MagicMock()
    pipe.delete = MagicMock()
    return client


@pytest.fixture
def cache(mock_redis):
    sc = SemanticCache(redis_url="redis://localhost:6379", similarity_threshold=0.92)
    sc._client = mock_redis
    return sc


class TestSemanticCacheGet:
    @pytest.mark.asyncio
    async def test_miss_on_empty_cache(self, cache, mock_redis):
        mock_redis.smembers.return_value = set()

        # Mock embedder to return a fixed vector
        with patch.object(cache, "_get_embedder") as mock_emb_factory:
            embedder = MagicMock()
            embedder.embed.return_value = [1.0] + [0.0] * 383
            mock_emb_factory.return_value = embedder

            result = await cache.get("hello world")
            assert result is None

    @pytest.mark.asyncio
    async def test_hit_above_threshold(self, cache, mock_redis):
        sha = "abc123"
        stored_emb = [1.0] + [0.0] * 383
        mock_redis.smembers.return_value = {sha}
        mock_redis.get.side_effect = lambda key: (
            json.dumps(stored_emb) if "emb" in key else "cached text"
        )

        with patch.object(cache, "_get_embedder") as mock_emb_factory:
            embedder = MagicMock()
            # Identical vector → similarity = 1.0 > 0.92
            embedder.embed.return_value = stored_emb
            mock_emb_factory.return_value = embedder

            result = await cache.get("hello world")
            assert result == "cached text"

    @pytest.mark.asyncio
    async def test_miss_below_threshold(self, cache, mock_redis):
        sha = "abc123"
        mock_redis.smembers.return_value = {sha}
        mock_redis.get.side_effect = lambda key: (
            json.dumps([1.0] + [0.0] * 383) if "emb" in key else "cached text"
        )

        with patch.object(cache, "_get_embedder") as mock_emb_factory:
            embedder = MagicMock()
            # Orthogonal vector → similarity = 0.0 < 0.92
            embedder.embed.return_value = [0.0, 1.0] + [0.0] * 382
            mock_emb_factory.return_value = embedder

            result = await cache.get("completely different")
            assert result is None

    @pytest.mark.asyncio
    async def test_exception_returns_none(self, cache, mock_redis):
        mock_redis.smembers.side_effect = Exception("Redis down")
        with patch.object(cache, "_get_embedder") as mock_emb_factory:
            embedder = MagicMock()
            embedder.embed.return_value = [1.0] + [0.0] * 383
            mock_emb_factory.return_value = embedder
            result = await cache.get("prompt")
            assert result is None


class TestSemanticCacheSet:
    @pytest.mark.asyncio
    async def test_set_stores_embedding_and_text(self, cache, mock_redis):
        pipe = mock_redis.pipeline.return_value

        with patch.object(cache, "_get_embedder") as mock_emb_factory:
            embedder = MagicMock()
            embedder.embed.return_value = [0.5] * 384
            mock_emb_factory.return_value = embedder

            await cache.set("my prompt", "my response")
            pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_exception_is_silent(self, cache, mock_redis):
        mock_redis.pipeline.side_effect = Exception("Redis down")
        with patch.object(cache, "_get_embedder") as mock_emb_factory:
            embedder = MagicMock()
            embedder.embed.return_value = [0.5] * 384
            mock_emb_factory.return_value = embedder
            # Should not raise
            await cache.set("prompt", "response")


class TestSemanticCacheClear:
    def test_clear_deletes_all_keys(self, cache, mock_redis):
        mock_redis.smembers.return_value = {"sha1", "sha2"}
        pipe = mock_redis.pipeline.return_value
        n = cache.clear()
        assert n == 2
        pipe.execute.assert_called_once()

    def test_clear_empty_cache_returns_zero(self, cache, mock_redis):
        mock_redis.smembers.return_value = set()
        assert cache.clear() == 0
