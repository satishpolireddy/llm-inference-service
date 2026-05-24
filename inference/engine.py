"""
InferenceEngine — unified abstraction over vLLM and HuggingFace backends.
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Selects and wraps the appropriate inference backend.

    Parameters
    ----------
    backend : str
        ``"vllm"`` for vLLM PagedAttention engine or ``"hf"`` for
        HuggingFace Transformers (CPU-safe fallback).
    model_name : str
        HuggingFace model identifier, e.g. ``"meta-llama/Llama-3.2-1B"``.
    """

    def __init__(self, backend: str = "hf", model_name: str | None = None) -> None:
        self.backend_name = backend
        self.model_name = model_name or os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B")
        self._backend = self._load_backend()

    def _load_backend(self):
        if self.backend_name == "vllm":
            from inference.vllm_backend import VLLMBackend
            logger.info("Loading vLLM backend for %s", self.model_name)
            return VLLMBackend(self.model_name)
        else:
            from inference.hf_backend import HFBackend
            logger.info("Loading HuggingFace backend for %s", self.model_name)
            return HFBackend(self.model_name)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> dict:
        """
        Synchronous generation.

        Returns
        -------
        dict with keys: text, tokens, inference_time, ttft_ms
        """
        start = time.perf_counter()
        result = self._backend.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )
        elapsed = time.perf_counter() - start
        result.setdefault("inference_time", elapsed)
        return result

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Async generator that yields tokens one-by-one."""
        async for token in self._backend.stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        ):
            yield token
