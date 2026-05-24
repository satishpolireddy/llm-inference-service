"""
HuggingFace Transformers backend — CPU-safe fallback inference engine.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class HFBackend:
    """
    CPU/GPU inference via HuggingFace ``transformers`` with ``TextIteratorStreamer``.

    This backend runs on any hardware — GPU is used automatically if available.
    For high-throughput GPU workloads, prefer ``VLLMBackend``.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._load()

    def _load(self) -> None:
        """Load model and tokenizer from HuggingFace Hub."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading tokenizer: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Loading model: %s", self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Model loaded on device: %s", next(self._model.parameters()).device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: list[str] | None = None,
    ) -> dict:
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        ttft_ms = (time.perf_counter() - t0) * 1000  # approximate for batch

        new_tokens = output_ids[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        inference_time = time.perf_counter() - t0

        return {
            "text": text,
            "tokens": len(new_tokens),
            "inference_time": inference_time,
            "ttft_ms": ttft_ms,
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
        import asyncio
        import threading

        import torch
        from transformers import TextIteratorStreamer

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            streamer=streamer,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        # Run generation in a background thread
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        loop = asyncio.get_event_loop()
        for token_text in streamer:
            if token_text:
                yield token_text
            await asyncio.sleep(0)  # yield control to event loop

        await loop.run_in_executor(None, thread.join)
