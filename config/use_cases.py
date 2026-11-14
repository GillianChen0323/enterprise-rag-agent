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
        "你是企业内部知识库智能助手。"
        "请严格基于以下检索到的文档内容回答问题，"
        "每条关键结论必须标注来源（如：[来源1]）。"
        "若文档中没有相关信息，请明确说明'知识库中暂无此信息'，不要猜测。"
        "\n\n检索文档：\n{context}"
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
        "你是智能客服辅助系统，帮助客服人员快速定位解决方案。"
        "请给出简洁、可操作的步骤，优先引用历史成功案例。"
        "回答格式：【问题分类】→【推荐方案】→【参考工单号】"
        "\n\n参考资料：\n{context}"
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
        "你是企业合规审查助手。"
        "请逐条分析以下文档中的潜在合规风险，"
        "对每个风险点标注：【风险等级：高/中/低】【涉及条款】【建议处理方式】。"
        "严格基于检索到的法规原文进行判断，不得推断。"
        "\n\n参考法规文档：\n{context}"
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
            f"未知的use_case: '{use_case}'。"
            f"支持的场景: {list(USE_CASE_REGISTRY.keys())}"
        )
    return USE_CASE_REGISTRY[use_case]
