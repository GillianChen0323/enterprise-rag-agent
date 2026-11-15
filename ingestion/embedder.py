"""Embedding client with optional Hypothetical Document Embeddings (HyDE). Generate a
hypothetical answer, then embed it with the query to align query and document
representations."""

import asyncio
import logging
from functools import lru_cache
from typing import Optional

import httpx
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Shared HTTP connection pool.
# ─────────────────────────────────────────────────────
_shared_http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=100,       # Maximum connections.
        max_keepalive_connections=20,  # Persistent connection count.
    ),
    timeout=httpx.Timeout(30.0),
)


class DomainEmbedder:
    """Embedding client supporting HyDE query expansion and batch document ingestion."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        hyde_model: str = "gpt-4o-mini",
        hyde_max_tokens: int = 150,
        dimensions: int = 1536,   # Supports reduced dimensions.
    ):
        self.model = model
        self.hyde_model = hyde_model
        self.hyde_max_tokens = hyde_max_tokens
        self.dimensions = dimensions
        self.client = AsyncOpenAI(
            api_key=api_key,
            http_client=_shared_http_client,  # Reuse connections.
        )

    # ─────────────────────────────────────────
    # Standard query embedding.
    # ─────────────────────────────────────────
    async def embed_query(self, query: str) -> list[float]:
        """Embed the original query without HyDE."""
        resp = await self.client.embeddings.create(
            model=self.model,
            input=query,
            dimensions=self.dimensions,
        )
        return resp.data[0].embedding

    # ─────────────────────────────────────────
    # HyDE expansion.
    # ─────────────────────────────────────────
    async def embed_query_with_hyde(self, query: str) -> list[float]:
        """Generate a hypothetical answer, combine it with the query, and embed the
        resulting text. Generation adds latency; use_hyde=False skips this step."""
        # First generate a hypothetical answer.
        # Embedding must wait for the generated answer.
        # Short generation limits extra latency.
        hypothesis = await self._generate_hypothesis(query)

        # Combine query and hypothetical answer.
        enriched_input = f"Question:{query}\n\nReference answer:{hypothesis}"

        # Embed combined text.
        resp = await self.client.embeddings.create(
            model=self.model,
            input=enriched_input,
            dimensions=self.dimensions,
        )
        logger.debug(f"HyDE embedding done | query={query[:40]} | "
                     f"hypothesis_len={len(hypothesis)}")
        return resp.data[0].embedding

    async def _generate_hypothesis(self, query: str) -> str:
        """Generate a short hypothetical answer for semantic guidance using gpt-4o-mini."""
        resp = await self.client.chat.completions.create(
            model=self.hyde_model,
            messages=[{
                "role": "user",
                "content": (
                    f"Briefly answer the following question in 2-3 sentences (for search optimization only; accuracy is not required):\n{query}"
                )
            }],
            max_tokens=self.hyde_max_tokens,
            temperature=0.3,    # Allow hypothetical answer diversity.
        )
        return resp.choices[0].message.content or ""

    # ─────────────────────────────────────────
    # Batch ingestion embeddings.
    # ─────────────────────────────────────────
    async def embed_documents_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Embed documents in batches during ingestion to reduce API call count."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = await self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
            )
            # Preserve input order.
            batch_embeddings = [
                e.embedding for e in sorted(resp.data, key=lambda x: x.index)
            ]
            all_embeddings.extend(batch_embeddings)
            logger.debug(f"Embedded batch {i//batch_size + 1} | "
                         f"{len(batch_embeddings)} docs")

        return all_embeddings

    # ─────────────────────────────────────────
    # Embed child chunks.
    # ─────────────────────────────────────────
    async def embed_document(self, text: str) -> list[float]:
        """Embed one document for an individual update."""
        resp = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )
        return resp.data[0].embedding
