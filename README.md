# LLM Inference Service

> High-throughput, real-time LLM inference with token streaming, Redis semantic caching, Prometheus observability, and Kubernetes-ready deployment.

[![CI](https://github.com/satishpolireddy/llm-inference-service/actions/workflows/ci.yml/badge.svg)](https://github.com/satishpolireddy/llm-inference-service/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Architecture

```
Client
  │
  ▼
FastAPI Gateway (8000)
  ├── POST /v1/generate          ← full response (JSON)
  ├── POST /v1/generate/stream   ← token-by-token SSE stream
  ├── GET  /v1/models            ← list loaded models
  ├── GET  /metrics              ← Prometheus metrics
  └── GET  /health
        │
        ├── Redis Cache ──────── semantic similarity lookup
        │       (6379)           skip inference on cache hit
        │
        └── Inference Backend
              ├── vLLM engine    (GPU, PagedAttention)
              └── HuggingFace    (CPU fallback)
```

## Key Features

| Feature | Detail |
|---|---|
| **Token streaming** | Server-Sent Events — first token < 200 ms |
| **Semantic cache** | Redis + cosine similarity; ~10× speedup on cache hits |
| **vLLM backend** | PagedAttention, continuous batching, 3–4× throughput vs naïve |
| **Benchmarking** | Automated TPS / TTFT / P95 latency reports |
| **Observability** | Prometheus counters + histograms; Grafana dashboard JSON |
| **Kubernetes** | Deployment + HPA + Service manifests |

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/satishpolireddy/llm-inference-service
cd llm-inference-service
cp .env.example .env          # set HF_TOKEN, MODEL_NAME, etc.

# 2. Start with Docker Compose
docker compose up --build

# 3. Stream a completion
curl -N -X POST http://localhost:8000/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformers in simple terms", "max_tokens": 200}'
```

## Benchmarks

Run the built-in benchmark suite:

```bash
python benchmarks/run_benchmark.py --concurrency 1 4 8 16 --prompt-file benchmarks/prompts.txt
```

Sample results (A100, Llama-3-8B):

| Concurrency | Throughput (tok/s) | TTFT P50 (ms) | TTFT P95 (ms) |
|---|---|---|---|
| 1 | 142 | 87 | 112 |
| 4 | 498 | 95 | 180 |
| 8 | 871 | 110 | 290 |
| 16 | 1 243 | 140 | 510 |

## Project Structure

```
llm-inference-service/
├── api/
│   ├── app.py            # FastAPI application factory
│   ├── routes.py         # /generate, /generate/stream, /models, /health
│   └── schemas.py        # Pydantic request/response models
├── inference/
│   ├── engine.py         # InferenceEngine abstraction
│   ├── vllm_backend.py   # vLLM PagedAttention backend
│   └── hf_backend.py     # HuggingFace Transformers fallback
├── cache/
│   ├── redis_cache.py    # Semantic cache with Redis + embeddings
│   └── embedder.py       # Lightweight embedding for cache keys
├── benchmarks/
│   ├── run_benchmark.py  # Async benchmark harness
│   └── prompts.txt       # Standard prompt set
├── monitoring/
│   ├── metrics.py        # Prometheus counters/histograms
│   └── grafana_dashboard.json
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── tests/
│   ├── test_api.py
│   ├── test_cache.py
│   └── test_inference.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `meta-llama/Llama-3.2-1B` | HuggingFace model ID |
| `BACKEND` | `hf` | `vllm` or `hf` |
| `MAX_TOKENS` | `512` | Default max new tokens |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `CACHE_SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity for cache hit |
| `ENABLE_METRICS` | `true` | Expose Prometheus `/metrics` |
| `HF_TOKEN` | — | HuggingFace API token |

## License

MIT
