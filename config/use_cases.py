"""Agent configurations for three business use cases."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Configurable agent settings. Each use case has its own AgentConfig and shares the
    RAG and LLM framework."""
    # Use-case identifier.
    use_case: str

    # Retrieval settings.
    top_k: int = 5                     # Final document count.
    candidate_k: int = 20              # Stage 1 candidate count.
    rerank: bool = True                # Enable Cohere reranking.
    score_threshold: float = 0.5       # Relevance threshold.
    hybrid_search: bool = True         # Enable hybrid retrieval.

    # Ingestion chunking settings.
    chunk_size: int = 512
    chunk_overlap: int = 64
    use_hierarchical_chunking: bool = True

    # Embedding settings.
    embedding_model: str = "text-embedding-3-large"
    use_hyde: bool = True              # HyDE query expansion.
    hyde_max_tokens: int = 150

    # LLM settings.
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.1           # Reduce generation variability.
    stream: bool = True

    # Cache settings.
    cache_ttl: int = 300               # TTL in seconds.
    cache_enabled: bool = True

    # Use-case-specific system prompt.
    system_prompt: str = ""

    # Tenant isolation field.
    tenant_filter_field: str = "tenant_id"


# ─────────────────────────────────────────────
# Three enterprise use cases.
# ─────────────────────────────────────────────

KB_QA_CONFIG = AgentConfig(
    use_case="kb_qa",
    top_k=5,
    candidate_k=20,
    rerank=True,
    score_threshold=0.5,
    hybrid_search=True,
    chunk_size=512,
    chunk_overlap=64,
    use_hierarchical_chunking=True,
    use_hyde=True,
    cache_ttl=600,          # Cache stable knowledge-base content longer.
    system_prompt=(
        "You are an internal enterprise knowledge-base assistant. "
        "Answer strictly based on the retrieved documents below. "
        "Cite the source for every key conclusion (for example, [Source1]). "
        "If the documents contain no relevant information, state 'This information is not available in the knowledge base.' Do not guess."
        "\n\nRetrieved documents:\n{context}"
    ),
)

HELPDESK_CONFIG = AgentConfig(
    use_case="helpdesk",
    top_k=3,
    candidate_k=15,
    rerank=True,
    score_threshold=0.55,   # Higher support relevance threshold.
    hybrid_search=True,
    chunk_size=256,          # Smaller support chunks.
    chunk_overlap=32,
    use_hierarchical_chunking=False,
    use_hyde=False,          # Skip HyDE for explicit support queries.
    cache_ttl=180,           # Short TTL for time-sensitive tickets.
    max_tokens=512,          # Concise support answers.
    system_prompt=(
        "You are a support assistant helping customer service staff quickly find solutions. "
        "Provide concise, actionable steps and prioritize references to previously successful cases. "
        "Response format: [Issue category] -> [Recommended solution] -> [Reference ticket number]"
        "\n\nReference material:\n{context}"
    ),
)

COMPLIANCE_CONFIG = AgentConfig(
    use_case="compliance",
    top_k=8,
    candidate_k=25,
    rerank=False,            # Broad compliance coverage.
    score_threshold=0.4,     # Retain potentially relevant clauses.
    hybrid_search=True,
    chunk_size=1024,         # Preserve longer compliance context.
    chunk_overlap=128,
    use_hierarchical_chunking=True,
    use_hyde=True,
    cache_ttl=3600,          # Long TTL for stable regulations.
    max_tokens=2048,
    temperature=0.0,         # Reduce output variability.
    system_prompt=(
        "You are an enterprise compliance review assistant. "
        "Analyze potential compliance risks in the following documents one by one. "
        "For each risk, specify: [Risk level: High/Medium/Low] [Relevant clauses] [Recommended action]. "
        "Base judgments strictly on the retrieved regulatory text; do not infer unsupported information."
        "\n\nReference regulatory documents:\n{context}"
    ),
)

# Configuration registry.
USE_CASE_REGISTRY: dict[str, AgentConfig] = {
    "kb_qa":       KB_QA_CONFIG,
    "helpdesk":    HELPDESK_CONFIG,
    "compliance":  COMPLIANCE_CONFIG,
}


def get_config(use_case: str) -> AgentConfig:
    """Return the configuration for a use-case name; raise ValueError if it is unknown."""
    if use_case not in USE_CASE_REGISTRY:
        raise ValueError(
            f"Unknown use_case: '{use_case}'; "
            f"Supported use cases: {list(USE_CASE_REGISTRY.keys())}"
        )
    return USE_CASE_REGISTRY[use_case]
