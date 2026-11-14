"""
agent/factory.py
----------------
AgentFactory: dependency-injection factory that assembles agents by use_case.
Reuse cached clients and connection pools instead of recreating them.
"""

import os
from functools import lru_cache
from config.use_cases import AgentConfig, get_config
from agent.core import EnterpriseRAGAgent
from retrieval.pipeline import OptimizedRAGPipeline
from retrieval.reranker import TwoStageRetriever
from ingestion.embedder import DomainEmbedder
from cache.semantic_cache import SemanticCache
from llm.client import LLMClient


# ─────────────────────────────────────────────
# Shared cached components to avoid recreating connection pools
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_shared_embedder() -> DomainEmbedder:
    return DomainEmbedder(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
    )


@lru_cache(maxsize=1)
def get_shared_cache() -> SemanticCache:
    return SemanticCache(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    )


@lru_cache(maxsize=1)
def get_shared_llm_client() -> LLMClient:
    return LLMClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        max_connections=100,
        max_keepalive=20,
    )


@lru_cache(maxsize=1)
def get_shared_reranker() -> TwoStageRetriever:
    return TwoStageRetriever(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )


# ─────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────

class AgentFactory:
    """
    Assemble an EnterpriseRAGAgent from a use_case string.
    Share underlying clients; instantiate AgentConfig and Pipeline per use case.
    """

    @staticmethod
    def create(use_case: str, qdrant_url: str = None) -> EnterpriseRAGAgent:
        """
        use_case: "kb_qa" | "helpdesk" | "compliance"
        """
        config: AgentConfig = get_config(use_case)

        # Connect to Qdrant; each use case has a separate collection.
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(
            url=qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        )
        collection_name = f"enterprise_{use_case}"

        # Retrieval pipeline containing the embedder and reranker
        pipeline = OptimizedRAGPipeline(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            embedder=get_shared_embedder(),
            reranker=get_shared_reranker(),
        )

        return EnterpriseRAGAgent(
            config=config,
            retriever=pipeline,
            llm_client=get_shared_llm_client(),
            cache=get_shared_cache(),
        )

    @staticmethod
    def create_all() -> dict[str, EnterpriseRAGAgent]:
        """Initialize agents for all use cases during service startup."""
        return {
            use_case: AgentFactory.create(use_case)
            for use_case in ["kb_qa", "helpdesk", "compliance"]
        }
