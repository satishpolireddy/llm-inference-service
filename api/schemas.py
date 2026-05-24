"""
Pydantic request / response schemas for the LLM Inference Service API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for the model.")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate.")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    top_p: float = Field(0.95, ge=0.0, le=1.0, description="Nucleus sampling probability.")
    top_k: int = Field(50, ge=0, description="Top-k sampling.")
    stop: list[str] | None = Field(None, description="Stop sequences.")
    stream: bool = Field(False, description="If true, stream tokens via SSE.")
    use_cache: bool = Field(True, description="Allow semantic cache lookup.")
    model: str | None = Field(None, description="Override the default model.")


class GenerateResponse(BaseModel):
    id: str
    model: str
    prompt: str
    text: str
    tokens_generated: int
    tokens_per_second: float
    time_to_first_token_ms: float
    total_time_ms: float
    cached: bool = False


class StreamChunk(BaseModel):
    id: str
    token: str
    index: int
    finish_reason: str | None = None


class ModelInfo(BaseModel):
    id: str
    backend: str
    loaded: bool
    context_length: int
    parameters: dict[str, Any] = {}


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
