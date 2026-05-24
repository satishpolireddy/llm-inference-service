"""
Prometheus metrics for the LLM Inference Service.

Metrics exposed at GET /metrics
--------------------------------
llm_requests_total              Counter   — total requests (by status)
llm_tokens_generated_total      Counter   — total tokens produced
llm_cache_hits_total            Counter   — semantic cache hits
llm_cache_misses_total          Counter   — semantic cache misses
llm_request_latency_seconds     Histogram — end-to-end request latency
llm_ttft_seconds                Histogram — time-to-first-token
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, make_asgi_app
from starlette.routing import Mount

# ── Counters ─────────────────────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total number of inference requests",
    ["status"],
)

TOKENS_GENERATED = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated across all requests",
)

CACHE_HITS = Counter(
    "llm_cache_hits_total",
    "Semantic cache hits",
)

CACHE_MISSES = Counter(
    "llm_cache_misses_total",
    "Semantic cache misses",
)

# ── Histograms ────────────────────────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "End-to-end request latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

TTFT_HISTOGRAM = Histogram(
    "llm_ttft_seconds",
    "Time to first token in seconds",
    buckets=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)


def setup_metrics(app) -> None:
    """Mount the Prometheus /metrics endpoint on the FastAPI app."""
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
