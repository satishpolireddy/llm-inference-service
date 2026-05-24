"""
Integration tests for the FastAPI endpoints.

Run with:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Create a test client with mocked inference engine and cache."""
    with (
        patch("api.routes._engine", create=True) as mock_engine,
        patch("api.routes._cache", create=True) as mock_cache,
    ):
        mock_engine.generate = AsyncMock(
            return_value={
                "text": "Hello, world!",
                "tokens_generated": 4,
                "time_to_first_token_ms": 50.0,
                "total_time_ms": 200.0,
            }
        )

        async def _fake_stream(prompt, **kw):
            for token in ["Hello", ",", " world", "!"]:
                yield token

        mock_engine.stream = _fake_stream
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()

        from api.app import create_app

        app = create_app()
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client):
        resp = client.get("/health")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# /v1/generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_basic_generate(self, client):
        resp = client.post(
            "/v1/generate",
            json={"prompt": "What is 2+2?", "max_tokens": 64},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert "tokens_generated" in data
        assert data["cached"] is False

    def test_generate_requires_prompt(self, client):
        resp = client.post("/v1/generate", json={"max_tokens": 64})
        assert resp.status_code == 422  # Unprocessable Entity

    def test_generate_max_tokens_bounds(self, client):
        # max_tokens must be between 1 and 4096
        resp = client.post(
            "/v1/generate", json={"prompt": "hi", "max_tokens": 0}
        )
        assert resp.status_code == 422

        resp = client.post(
            "/v1/generate", json={"prompt": "hi", "max_tokens": 9999}
        )
        assert resp.status_code == 422

    def test_generate_temperature_bounds(self, client):
        resp = client.post(
            "/v1/generate", json={"prompt": "hi", "temperature": -0.1}
        )
        assert resp.status_code == 422

    def test_generate_cache_hit(self, client):
        with patch("api.routes._cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value="cached response")
            resp = client.post(
                "/v1/generate",
                json={"prompt": "cached prompt", "use_cache": True},
            )
            assert resp.status_code == 200
            assert resp.json()["cached"] is True

    def test_generate_cache_bypass(self, client):
        with patch("api.routes._cache") as mock_cache:
            mock_cache.get = AsyncMock(return_value="cached response")
            resp = client.post(
                "/v1/generate",
                json={"prompt": "skip cache", "use_cache": False},
            )
            # Should not use cache
            mock_cache.get.assert_not_called()


# ---------------------------------------------------------------------------
# /v1/generate/stream
# ---------------------------------------------------------------------------

class TestStream:
    def test_stream_returns_event_stream(self, client):
        resp = client.post(
            "/v1/generate/stream",
            json={"prompt": "Count to 3", "stream": True},
            stream=True,
        )
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct

    def test_stream_emits_data_events(self, client):
        resp = client.post(
            "/v1/generate/stream",
            json={"prompt": "Count to 3", "stream": True},
        )
        body = resp.text
        assert "data:" in body


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

class TestModels:
    def test_models_returns_list(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
