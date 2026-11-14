"""Cohere reranking for two-stage retrieval: retrieve candidates, rerank, and filter by
relevance score. Measure quality and groundedness on a representative evaluation
dataset."""

import asyncio
import logging
from typing import Optional
import cohere

from agent.core import Document

logger = logging.getLogger(__name__)


class TwoStageRetriever:
    """Rerank candidate documents and return top-K results with relevance-score metadata."""

    def __init__(
        self,
        cohere_api_key: str,
        model: str = "rerank-multilingual-v3.0",
    ):
        self.co = cohere.AsyncClient(api_key=cohere_api_key)
        self.model = model

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int = 5,
        score_threshold: float = 0.5,
    ) -> list[Document]:
        """Rerank candidates with Cohere and filter low-relevance results. High thresholds
        may reduce recall; low thresholds may retain noise. Tune on representative
        evaluation data."""
        if not documents:
            return []

        # Truncate candidate count.
        candidates = documents[:1000]
        docs_text = [d.page_content for d in candidates]

        try:
            resp = await self.co.rerank(
                model=self.model,
                query=query,
                documents=docs_text,
                top_n=min(top_n * 2, len(candidates)),  # Request extra results before filtering.
                return_documents=True,
            )
        except cohere.CohereAPIError as e:
            logger.error(f"Cohere rerank failed: {e}. Falling back to vector scores.")
            # Fallback to vector scores.
            return sorted(
                candidates,
                key=lambda d: d.metadata.get("score", 0),
                reverse=True
            )[:top_n]

        # Store and filter relevance scores.
        reranked = []
        for r in resp.results:
            if r.relevance_score < score_threshold:
                # Reduce low-relevance context noise.
                # Measure groundedness impact on evaluation data.
                logger.debug(f"Filtered doc (score={r.relevance_score:.3f} < {score_threshold})")
                continue

            doc = candidates[r.index]
            doc.metadata["relevance_score"] = r.relevance_score
            doc.metadata["rerank_position"] = len(reranked) + 1
            reranked.append(doc)

            if len(reranked) >= top_n:
                break

        logger.info(
            f"Rerank: {len(candidates)} candidates → "
            f"{len(reranked)} passed threshold={score_threshold}"
        )
        return reranked

    async def rerank_batch(
        self,
        queries: list[str],
        documents_list: list[list[Document]],
        top_n: int = 5,
        score_threshold: float = 0.5,
    ) -> list[list[Document]]:
        """Rerank multiple queries concurrently for batch evaluation."""
        tasks = [
            self.rerank(q, docs, top_n, score_threshold)
            for q, docs in zip(queries, documents_list)
        ]
        return await asyncio.gather(*tasks)
