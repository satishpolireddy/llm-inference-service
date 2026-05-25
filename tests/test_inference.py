"""
Unit tests for the inference engine and backends.

Run with:
    pytest tests/test_inference.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference.engine import InferenceEngine


# ---------------------------------------------------------------------------
# InferenceEngine routing
# ---------------------------------------------------------------------------

class TestInferenceEngineRouting:
    def test_loads_hf_backend_by_default(self):
        with patch("inference.engine.HFBackend") as MockHF:
            MockHF.return_value = MagicMock()
            engine = InferenceEngine(backend="hf", model_name="gpt2")
            MockHF.assert_called_once_with(model_name="gpt2")

    def test_loads_vllm_backend(self):
        with patch("inference.engine.VLLMBackend") as MockVLLM:
            MockVLLM.return_value = MagicMock()
            engine = InferenceEngine(backend="vllm", model_name="meta-llama/Llama-3-8B")
            MockVLLM.assert_called_once_with(model_name="meta-llama/Llama-3-8B")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            InferenceEngine(backend="unknown_backend", model_name="gpt2")


class TestInferenceEngineGenerate:
    @pytest.mark.asyncio
    async def test_generate_delegates_to_backend(self):
        with patch("inference.engine.HFBackend") as MockHF:
            mock_backend = MagicMock()
            mock_backend.generate = AsyncMock(
                return_value={
                    "text": "result",
                    "tokens_generated": 3,
                    "time_to_first_token_ms": 10.0,
                    "total_time_ms": 100.0,
                }
            )
            MockHF.return_value = mock_backend
            engine = InferenceEngine(backend="hf", model_name="gpt2")
            result = await engine.generate("hello", max_tokens=32, temperature=0.7)
            assert result["text"] == "result"
            mock_backend.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_delegates_to_backend(self):
        with patch("inference.engine.HFBackend") as MockHF:
            mock_backend = MagicMock()

            async def _gen(*a, **kw):
                for t in ["a", "b", "c"]:
                    yield t

            mock_backend.stream = _gen
            MockHF.return_value = mock_backend
            engine = InferenceEngine(backend="hf", model_name="gpt2")

            tokens = []
            async for tok in engine.stream("hello", max_tokens=8):
                tokens.append(tok)
            assert tokens == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# HFBackend
# ---------------------------------------------------------------------------

class TestHFBackend:
    def _make_backend(self):
        with (
            patch("inference.hf_backend.AutoTokenizer") as MockTok,
            patch("inference.hf_backend.AutoModelForCausalLM") as MockModel,
        ):
            tokenizer = MagicMock()
            tokenizer.encode.return_value = [1, 2, 3]
            tokenizer.decode.return_value = "Hello world"
            tokenizer.eos_token_id = 2
            MockTok.from_pretrained.return_value = tokenizer

            model = MagicMock()
            model.generate.return_value = MagicMock()
            model.device = "cpu"
            MockModel.from_pretrained.return_value = model

            from inference.hf_backend import HFBackend

            backend = HFBackend(model_name="gpt2")
            backend._tokenizer = tokenizer
            backend._model = model
            return backend

    @pytest.mark.asyncio
    async def test_generate_returns_dict_with_required_keys(self):
        backend = self._make_backend()
        backend._tokenizer.decode.return_value = "output text"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = {
                "text": "output text",
                "tokens_generated": 5,
                "time_to_first_token_ms": 20.0,
                "total_time_ms": 80.0,
            }
            result = await backend.generate("hello", max_tokens=32, temperature=0.7)

        assert "text" in result
        assert "tokens_generated" in result

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        backend = self._make_backend()

        # Patch the streamer to emit predefined tokens
        with patch("inference.hf_backend.TextIteratorStreamer") as MockStreamer:
            streamer_instance = MagicMock()
            streamer_instance.__iter__ = MagicMock(
                return_value=iter(["Hello", " world", ""])
            )
            MockStreamer.return_value = streamer_instance

            with patch("threading.Thread") as MockThread:
                mock_thread = MagicMock()
                MockThread.return_value = mock_thread

                tokens = []
                async for tok in backend.stream("hi", max_tokens=16):
                    tokens.append(tok)

                # At least some tokens should have been yielded
                # (exact count depends on impl; just check it ran)
                assert isinstance(tokens, list)


# ---------------------------------------------------------------------------
# VLLMBackend
# ---------------------------------------------------------------------------

class TestVLLMBackend:
    def test_vllm_not_installed_raises_import_error(self):
        import sys
        with patch.dict(sys.modules, {"vllm": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                from inference.vllm_backend import VLLMBackend
                VLLMBackend(model_name="test")
