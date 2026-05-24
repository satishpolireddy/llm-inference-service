"""
vLLM backend — PagedAttention-powered high-throughput inference.

Requires: pip install vllm
GPU with CUDA support is required for vLLM.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class VLLMBackend:
    """
    High-throughput inference backend using vLLM's PagedAttention engine.

    vLLM provides:
    - PagedAttention: near-zero KV-cache waste → higher batch sizes
    - Continuous batching: new requests slot into ongoing inference
    - Optimised CUDA kernels for attention and sampling

    Compared to naive HuggingFace generation, vLLM typically achieves
    3–4× higher throughput on A100-class hardware.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._llm = None
        self._load()

    def _load(self) -> None:
        try:
            from vllm import LLM, SamplingParams  # noqa: F401

            logger.info("Initialising vLLM engine for %s", self.model_name)
            self._llm = LLM(
                model=self.model_name,
                dtype="float16",
                max_model_len=4096,
                trust_remote_code=True,
            )
            logger.info("vLLM engine ready")
        except ImportError:
            raise RuntimeError(
                "vLLM is not installed. Install it with: pip install vllm\n"
                "Note: vLLM requires a CUDA-capable GPU."
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> dict:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )

        t0 = time.perf_counter()
        outputs = self._llm.generate([prompt], sampling_params)
        elapsed = time.perf_counter() - t0

        result = outputs[0]
        text = result.outputs[0].text
        tokens = len(result.outputs[0].token_ids)

        return {
            "text": text,
            "tokens": tokens,
            "inference_time": elapsed,
            "ttft_ms": elapsed * 1000 / max(tokens, 1),  # approximate
        }

    async def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens using vLLM's async engine.

        Uses ``AsyncLLMEngine`` for non-blocking per-token streaming.
        """
        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
        except ImportError:
            raise RuntimeError("vLLM not installed.")

        # Lazily create async engine for streaming
        if not hasattr(self, "_async_engine"):
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                dtype="float16",
                max_model_len=4096,
                trust_remote_code=True,
            )
            self._async_engine = AsyncLLMEngine.from_engine_args(engine_args)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop or [],
        )

        import uuid
        request_id = str(uuid.uuid4())
        results_generator = self._async_engine.generate(prompt, sampling_params, request_id)

        previous_text = ""
        async for request_output in results_generator:
            new_text = request_output.outputs[0].text
            delta = new_text[len(previous_text):]
            previous_text = new_text
            if delta:
                yield delta
