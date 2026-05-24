"""
API route definitions for the LLM Inference Service.

Endpoints
---------
POST /v1/generate           — synchronous generation (full response)
POST /v1/generate/stream    — token-by-token SSE stream
GET  /v1/models             — list available models
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import GenerateRequest, GenerateResponse, ModelInfo, ModelsResponse
from cache.redis_cache import SemanticCache
from inference.engine import InferenceEngine
from monitoring.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    REQUEST_LATENCY,
    TOKENS_GENERATED,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Singletons (initialised lazily on first request)
_engine: InferenceEngine | None = None
_cache: SemanticCache | None = None


def _get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        backend = os.getenv("BACKEND", "hf")
        model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
        _engine = InferenceEngine(backend=backend, model_name=model_name)
    return _engine


def _get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
        _cache = SemanticCache(redis_url=redis_url, similarity_threshold=threshold)
    return _cache


@router.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(body: GenerateRequest):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    cached_text = None
    if body.use_cache:
        try:
            cached_text = await _get_cache().get(body.prompt)
        except Exception as exc:
            logger.warning("Cache lookup failed: %s", exc)
    if cached_text is not None:
        CACHE_HITS.inc()
        total_ms = (time.perf_counter() - start_time) * 1000
        return GenerateResponse(id=request_id, model=os.getenv("MODEL_NAME","unknown"), prompt=body.prompt, text=cached_text, tokens_generated=len(cached_text.split()), tokens_per_second=0.0, time_to_first_token_ms=0.0, total_time_ms=round(total_ms,2), cached=True)
    CACHE_MISSES.inc()
    try:
        engine = _get_engine()
        result = await asyncio.to_thread(engine.generate, prompt=body.prompt, max_tokens=body.max_tokens, temperature=body.temperature, top_p=body.top_p, top_k=body.top_k, stop=body.stop)
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")
    total_ms = (time.perf_counter() - start_time) * 1000
    TOKENS_GENERATED.inc(result.get("tokens",0))
    REQUEST_LATENCY.observe(total_ms/1000)
    if body.use_cache and result.get("text"):
        try:
            await _get_cache().set(body.prompt, result["text"])
        except Exception as exc:
            logger.warning("Cache store failed: %s", exc)
    return GenerateResponse(id=request_id, model=os.getenv("MODEL_NAME","unknown"), prompt=body.prompt, text=result.get("text",""), tokens_generated=result.get("tokens",0), tokens_per_second=round(result.get("tokens",0)/max(result.get("inference_time",1),1e-6),2), time_to_first_token_ms=round(result.get("ttft_ms",0.0),2), total_time_ms=round(total_ms,2), cached=False)


@router.post("/generate/stream", tags=["inference"])
async def generate_stream(body: GenerateRequest):
    request_id = str(uuid.uuid4())
    async def event_generator():
        engine = _get_engine()
        index = 0
        try:
            async for token in engine.stream(prompt=body.prompt, max_tokens=body.max_tokens, temperature=body.temperature, top_p=body.top_p, top_k=body.top_k, stop=body.stop):
                yield f"data: {json.dumps({'id':request_id,'token':token,'index':index})}\n\n"
                index += 1
        except Exception as exc:
            yield f"data: {json.dumps({'id':request_id,'error':str(exc)})}\n\n"
        yield f"data: {json.dumps({'id':request_id,'token':'','index':index,'finish_reason':'stop'})}\n\n"
    return StreamingResponse(event_generator(),media_type="text/event-stream",headers={"Cache-Control":"no-cache"})


@router.get("/models", tags=["models"])
async def list_models():
    model_name = os.getenv("MODEL_NAME","meta-llama/Llama-3.2-1B")
    backend = os.getenv("BACKEND","hf")
    return ModelsResponse(models=[ModelInfo(id=model_name,backend=backend,loaded=True,context_length=4096,parameters={"dtype":"float16"})])
