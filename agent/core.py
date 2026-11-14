"""
agent/core.py
-------------
EnterpriseRAGAgent: the core agent connecting retrieval and generation.
Architecture: an LLM-based agent with RAG retrieval pipelines.
"""

import time
import logging
from typing import AsyncIterator, Optional

from config.use_cases import AgentConfig

logger = logging.getLogger(__name__)


class Document:
    """Lightweight document container; can be replaced with a LangChain or custom document type."""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class EnterpriseRAGAgent:
    """
    RAG agent integrating retrieval and generation.
    - Supports tenant-based filtering using tenant_id.
    - Supports streaming output.
    - Integrates semantic caching, a retrieval pipeline, and LLM generation.
    - Uses AgentConfig for configurable business logic.
    """

    def __init__(
        self,
        config: AgentConfig,
        retriever,       # OptimizedRAGPipeline instance
        llm_client,      # LLMClient instance
        cache,           # SemanticCache instance
    ):
        self.config = config
        self.retriever = retriever
        self.llm = llm_client
        self.cache = cache

    # ─────────────────────────────────────────
    # Non-streaming entry point
    # ─────────────────────────────────────────
    async def run(self, query: str, tenant_id: str) -> dict:
        """
        Full RAG flow: cache lookup -> retrieval -> generation -> cache write.
        Returns: {answer, sources, latency_ms, from_cache}
        """
        t0 = time.perf_counter()

        # Step 1: Check the cache and return cached results when available.
        if self.config.cache_enabled:
            cached = await self.cache.get(query, self.config.use_case, tenant_id)
            if cached:
                logger.info(f"[{self.config.use_case}] Cache hit | query={query[:50]}")
                return {**cached, "from_cache": True, "latency_ms": 0}

        # Step 2: RAG retrieval
        docs = await self.retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            rerank=self.config.rerank,
            score_threshold=self.config.score_threshold,
            use_hyde=self.config.use_hyde,
        )
        logger.info(f"[{self.config.use_case}] Retrieved {len(docs)} docs")

        # Step 3: Build the prompt context
        context = self._build_context(docs)
        system_prompt = self.config.system_prompt.format(context=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Step 4: Generate the response without streaming
        answer = await self.llm.generate(
            messages=messages,
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000)
        logger.info(f"[{self.config.use_case}] Done | latency={latency_ms}ms")

        result = {
            "answer": answer,
            "sources": self._format_sources(docs),
            "latency_ms": latency_ms,
            "use_case": self.config.use_case,
            "from_cache": False,
        }

        # Step 5: Write the result to the cache
        if self.config.cache_enabled:
            await self.cache.set(
                query, self.config.use_case, tenant_id,
                result, ttl=self.config.cache_ttl
            )

        return result

    # ─────────────────────────────────────────
    # Streaming entry point (Server-Sent Events)
    # ─────────────────────────────────────────
    async def run_stream(
        self, query: str, tenant_id: str
    ) -> AsyncIterator[str]:
        """
        Streaming RAG: start streaming generation after retrieval completes.
        Usage: async for token in agent.run_stream(query, tenant_id)
        """
        # Retrieve documents before streaming, as above.
        docs = await self.retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            rerank=self.config.rerank,
            score_threshold=self.config.score_threshold,
            use_hyde=self.config.use_hyde,
        )

        context = self._build_context(docs)
        system_prompt = self.config.system_prompt.format(context=context)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Stream the LLM response
        async for token in self.llm.stream_generate(
            messages=messages,
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ):
            yield token

    # ─────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────
    def _build_context(self, docs: list[Document]) -> str:
        """
        Combine retrieved documents into a structured context string.
        With hierarchical chunking, use parent chunk content for broader context.
        """
        if not docs:
            return "（未检索到相关文档）"
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            header = f"[来源{i}] {source}" + (f" | 第{page}页" if page else "")
            parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _format_sources(self, docs: list[Document]) -> list[dict]:
        """Return structured sources for rendering citations in the frontend."""
        return [
            {
                "index": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", ""),
                "score": doc.metadata.get("relevance_score", None),
            }
            for i, doc in enumerate(docs)
        ]
