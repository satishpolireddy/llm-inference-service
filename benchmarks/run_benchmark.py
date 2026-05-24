"""
Async benchmark harness for the LLM Inference Service.

Measures:
  - Tokens per second (TPS)
  - Time to first token (TTFT)
  - End-to-end latency (P50, P95, P99)

Usage
-----
    python benchmarks/run_benchmark.py \
        --url http://localhost:8000/v1 \
        --concurrency 1 4 8 16 \
        --prompts benchmarks/prompts.txt \
        --max-tokens 256
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import NamedTuple

import aiohttp


class Result(NamedTuple):
    prompt_idx: int
    concurrency: int
    ttft_s: float          # time-to-first-token in seconds
    total_s: float         # total wall-clock time
    tokens_generated: int
    tps: float             # tokens per second


async def run_one(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    prompt_idx: int,
    concurrency: int,
    max_tokens: int,
) -> Result:
    url = f"{base_url}/generate/stream"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "use_cache": False,
    }

    t0 = time.perf_counter()
    ttft_s = 0.0
    tokens = 0
    first_token = True

    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if first_token:
                ttft_s = time.perf_counter() - t0
                first_token = False
            tokens += len(chunk.get("token", "").split())

    total_s = time.perf_counter() - t0
    tps = tokens / total_s if total_s > 0 else 0.0
    return Result(prompt_idx, concurrency, ttft_s, total_s, tokens, tps)


async def benchmark_concurrency(
    base_url: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
) -> list[Result]:
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timeout = aiohttp.ClientTimeout(total=120)
    results: list[Result] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warm-up: single request
        try:
            await run_one(session, base_url, prompts[0], -1, concurrency, max_tokens)
        except Exception:
            pass

        sem = asyncio.Semaphore(concurrency)

        async def bounded(idx: int, prompt: str) -> Result | None:
            async with sem:
                try:
                    return await run_one(
                        session, base_url, prompt, idx, concurrency, max_tokens
                    )
                except Exception as exc:
                    print(f"  [warn] prompt {idx} failed: {exc}")
                    return None

        tasks = [bounded(i, p) for i, p in enumerate(prompts)]
        raw = await asyncio.gather(*tasks)
        results = [r for r in raw if r is not None]

    return results


def print_report(concurrency: int, results: list[Result]) -> None:
    if not results:
        print(f"  concurrency={concurrency}: no results")
        return

    ttfts = [r.ttft_s * 1000 for r in results]         # ms
    totals = [r.total_s * 1000 for r in results]        # ms
    tps_vals = [r.tps for r in results]

    def pct(data: list[float], p: int) -> float:
        data_sorted = sorted(data)
        idx = max(0, int(len(data_sorted) * p / 100) - 1)
        return data_sorted[idx]

    print(f"\n  concurrency={concurrency}  n={len(results)}")
    print(f"    TTFT   — mean={statistics.mean(ttfts):.1f}ms  "
          f"p50={pct(ttfts,50):.1f}ms  "
          f"p95={pct(ttfts,95):.1f}ms  "
          f"p99={pct(ttfts,99):.1f}ms")
    print(f"    Latency— mean={statistics.mean(totals):.1f}ms  "
          f"p50={pct(totals,50):.1f}ms  "
          f"p95={pct(totals,95):.1f}ms  "
          f"p99={pct(totals,99):.1f}ms")
    print(f"    TPS    — mean={statistics.mean(tps_vals):.1f}  "
          f"p50={pct(tps_vals,50):.1f}  "
          f"max={max(tps_vals):.1f}")


async def main(args: argparse.Namespace) -> None:
    prompts_path = Path(args.prompts)
    prompts = [
        line.strip()
        for line in prompts_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not prompts:
        print("No prompts found. Exiting.")
        return

    print(f"Benchmark: {len(prompts)} prompts, max_tokens={args.max_tokens}")
    print(f"Target: {args.url}")
    print("=" * 60)

    for c in args.concurrency:
        results = await benchmark_concurrency(
            args.url, prompts, concurrency=c, max_tokens=args.max_tokens
        )
        print_report(c, results)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Service benchmark")
    parser.add_argument("--url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--concurrency", nargs="+", type=int, default=[1, 4, 8, 16]
    )
    parser.add_argument("--prompts", default="benchmarks/prompts.txt")
    parser.add_argument("--max-tokens", type=int, default=256)
    asyncio.run(main(parser.parse_args()))
