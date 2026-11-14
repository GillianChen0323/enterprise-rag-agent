"""Asynchronous retrieval pipeline with parallel dense and sparse branches, optional Cohere
reranking, and HyDE query expansion. The sparse branch is a placeholder in this
repository."""

import asyncio
import logging
import time
from typing import Optional

from agent.core import Document

logger = logging.getLogger(__name__)


class OptimizedRAGPipeline:
    """Run dense retrieval and a sparse retrieval hook concurrently, merge results, and
    optionally rerank. The provided BM25 hook is a mock implementation."""

    def __init__(self, qdrant_client, collection_name: str, embedder, reranker):
        self.qdrant = qdrant_client
        self.collection = collection_name
        self.embedder = embedder
        self.reranker = reranker

        # Collect latency metrics.
        self._latency_records: list[float] = []

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        candidate_k: int = 20,
        rerank: bool = True,
        score_threshold: float = 0.5,
        use_hyde: bool = True,
    ) -> list[Document]:
        """Run parallel retrieval, merge and deduplicate candidates, optionally rerank, and
        filter by relevance threshold."""
        t0 = time.perf_counter()

        # Parallel retrieval.
        # Sequential comparison.
        # dense_docs  = await self._dense_search(query, tenant_id, candidate_k, use_hyde)
        # sparse_docs = await self._sparse_search(query, tenant_id, candidate_k)

        # Run both branches with asyncio.gather.
        dense_task  = asyncio.create_task(
            self._dense_search(query, tenant_id, candidate_k, use_hyde)
        )
        sparse_task = asyncio.create_task(
            self._sparse_search(query, tenant_id, candidate_k)
        )
        dense_docs, sparse_docs = await asyncio.gather(dense_task, sparse_task)

        t_retrieval = (time.perf_counter() - t0) * 1000
        logger.debug(f"Parallel retrieval done in {t_retrieval:.1f}ms | "
                     f"dense={len(dense_docs)}, sparse={len(sparse_docs)}")

        # Merge and deduplicate.
        merged = self._deduplicate(dense_docs + sparse_docs)

        if not merged:
            logger.warning(f"No documents retrieved for query: {query[:60]}")
            return []

        # Optional reranking.
        if rerank and len(merged) > top_k:
            docs = await self.reranker.rerank(
                query=query,
                documents=merged,
                top_n=top_k,
                score_threshold=score_threshold,
            )
        else:
            # Apply score threshold without reranking too.
            docs = [
                d for d in merged[:top_k]
                if d.metadata.get("score", 1.0) >= score_threshold
            ]

        total_ms = (time.perf_counter() - t0) * 1000
        self._latency_records.append(total_ms)
        logger.info(f"Retrieval complete | final_docs={len(docs)} | "
                    f"total={total_ms:.1f}ms")
        return docs

    # ─────────────────────────────────────────────
    # Dense retrieval.
    # ─────────────────────────────────────────────
    async def _dense_search(
        self,
        query: str,
        tenant_id: str,
        k: int,
        use_hyde: bool,
    ) -> list[Document]:
        """Retrieve using query embeddings, optionally expanded with a hypothetical answer
        through HyDE."""
        # Query embedding with optional HyDE.
        if use_hyde:
            query_vector = await self.embedder.embed_query_with_hyde(query)
        else:
            query_vector = await self.embedder.embed_query(query)

        # Tenant-filtered Qdrant retrieval.
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = await asyncio.to_thread(
            self.qdrant.search,
            collection_name=self.collection,
            query_vector=query_vector,
            limit=k,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            ),
            with_payload=True,
        )

        docs = []
        for hit in results:
            doc = Document(
                page_content=hit.payload.get("parent_text") or hit.payload.get("text", ""),
                metadata={
                    "source": hit.payload.get("source", ""),
                    "page": hit.payload.get("page", ""),
                    "score": hit.score,
                    "chunk_id": hit.id,
                    "tenant_id": tenant_id,
                    "retrieval_type": "dense",
                }
            )
            docs.append(doc)
        return docs

    # ─────────────────────────────────────────────
    # Sparse retrieval.
    # ─────────────────────────────────────────────
    async def _sparse_search(
        self,
        query: str,
        tenant_id: str,
        k: int,
    ) -> list[Document]:
        """Placeholder for sparse retrieval. Replace with Elasticsearch BM25 or Qdrant
        sparse vectors for actual keyword retrieval. The mock illustrates orchestration
        and returns no documents."""
        # Replace with a real sparse retrieval backend.
        # Placeholder for keyword retrieval.
        await asyncio.sleep(0.0)  # Replace with a real BM25 call.

        # Mock returns an empty list.
        # results = await self.es_client.search(
        #     index=f"enterprise_{tenant_id}",
        #     body={"query": {"match": {"content": query}}, "size": k}
        # )
        return []

    # ─────────────────────────────────────────────
    # Content-based deduplication.
    # ─────────────────────────────────────────────
    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        """Deduplicate by the first 200 characters of content, retaining the first
        occurrence; dense results come first."""
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    # ─────────────────────────────────────────────
    # Latency statistics.
    # ─────────────────────────────────────────────
    def get_latency_stats(self) -> dict:
        """Return P50/P95/P99 retrieval latency statistics."""
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
