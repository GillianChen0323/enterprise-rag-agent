"""LLM client with a shared HTTP connection pool, streaming generation, and exponential-
backoff retries for rate limits and server errors."""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Process-wide connection pool.
# ─────────────────────────────────────────────────────
# Per-request clients repeat connection setup.
# Reuse persistent connections.
_SHARED_HTTP_CLIENT = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,           # Concurrent connection limit.
        max_keepalive_connections=20,  # Persistent connection limit.
    ),
    timeout=httpx.Timeout(
        connect=5.0,   # Connection timeout.
        read=30.0,     # Streaming read timeout.
        write=10.0,
        pool=5.0,      # Pool wait timeout.
    ),
)


class LLMClient:
    """LLM client supporting pooled connections, complete responses, streaming responses,
    and retries."""

    def __init__(
        self,
        api_key: str,
        max_connections: int = 100,
        max_keepalive: int = 20,
        max_retries: int = 3,
    ):
        self.max_retries = max_retries
        self.client = AsyncOpenAI(
            api_key=api_key,
            http_client=_SHARED_HTTP_CLIENT,
            max_retries=0,  # Explicit retry handling.
        )
        # Record generation latency.
        self._latency_records: list[float] = []

    # ─────────────────────────────────────────
    # Complete response generation.
    # ─────────────────────────────────────────
    async def generate(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Return a complete answer string with exponential-backoff retries for rate
        limits."""
        t0 = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                answer = resp.choices[0].message.content or ""

                latency_ms = (time.perf_counter() - t0) * 1000
                self._latency_records.append(latency_ms)
                logger.debug(f"LLM generate done | model={model} | "
                             f"tokens={resp.usage.completion_tokens} | "
                             f"latency={latency_ms:.0f}ms")
                return answer

            except RateLimitError as e:
                wait = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds.
                logger.warning(f"Rate limit hit (attempt {attempt+1}), "
                               f"retrying in {wait}s...")
                await asyncio.sleep(wait)
                last_error = e

            except APIStatusError as e:
                if e.status_code >= 500:  # Retry server errors.
                    wait = 2 ** attempt
                    logger.warning(f"OpenAI 5xx (attempt {attempt+1}), "
                                   f"retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    last_error = e
                else:
                    raise  # Do not retry other client errors.

        raise RuntimeError(f"LLM generate failed after {self.max_retries} retries: {last_error}")

    # ─────────────────────────────────────────
    # Streaming generation.
    # ─────────────────────────────────────────
    async def stream_generate(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        """Yield generated tokens incrementally. Target time to first token: < 500 ms.
        Consume with async for token in llm.stream_generate(messages)."""
        t_start = time.perf_counter()
        first_token_logged = False

        async with self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ) as stream:
            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    if not first_token_logged:
                        ttft_ms = (time.perf_counter() - t_start) * 1000
                        logger.debug(f"Time to first token: {ttft_ms:.0f}ms")
                        first_token_logged = True
                    yield token

    # ─────────────────────────────────────────
    # Latency metrics.
    # ─────────────────────────────────────────
    def get_latency_stats(self) -> dict:
        if not self._latency_records:
            return {}
        import numpy as np
        arr = np.array(self._latency_records)
        return {
            "p50_ms":  round(float(np.percentile(arr, 50)), 1),
            "p95_ms":  round(float(np.percentile(arr, 95)), 1),
            "p99_ms":  round(float(np.percentile(arr, 99)), 1),
            "mean_ms": round(float(arr.mean()), 1),
            "count":   len(arr),
        }
