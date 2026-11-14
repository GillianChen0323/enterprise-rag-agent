"""
config/use_cases.py
-------------------
3个业务场景的差异化AgentConfig配置。
对应简历：configurable business logic, serving enterprise clients across 3 distinct use cases
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """
    可配置的Agent业务逻辑参数。
    每个enterprise use case对应一个AgentConfig实例，
    共用同一套RAG + LLM框架，通过配置驱动差异化行为。
    """
    
    use_case: str

    
    top_k: int = 5                     
    candidate_k: int = 20              
    rerank: bool = True                
    score_threshold: float = 0.5       
    hybrid_search: bool = True         

    
    chunk_size: int = 512
    chunk_overlap: int = 64
    use_hierarchical_chunking: bool = True

    
    embedding_model: str = "text-embedding-3-large"
    use_hyde: bool = True              
    hyde_max_tokens: int = 150

    
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.1           
    stream: bool = True

    
    cache_ttl: int = 300               
    cache_enabled: bool = True

    
    system_prompt: str = ""

    
    tenant_filter_field: str = "tenant_id"


# ─────────────────────────────────────────────

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
    cache_ttl=600,          
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
    score_threshold=0.55,   
    hybrid_search=True,
    chunk_size=256,          
    chunk_overlap=32,
    use_hierarchical_chunking=False,
    use_hyde=False,          
    cache_ttl=180,           
    max_tokens=512,          
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
    rerank=False,            
    score_threshold=0.4,     
    hybrid_search=True,
    chunk_size=1024,         
    chunk_overlap=128,
    use_hierarchical_chunking=True,
    use_hyde=True,
    cache_ttl=3600,          
    max_tokens=2048,
    temperature=0.0,         
    system_prompt=(
        "你是企业合规审查助手。"
        "请逐条分析以下文档中的潜在合规风险，"
        "对每个风险点标注：【风险等级：高/中/低】【涉及条款】【建议处理方式】。"
        "严格基于检索到的法规原文进行判断，不得推断。"
        "\n\n参考法规文档：\n{context}"
    ),
)


USE_CASE_REGISTRY: dict[str, AgentConfig] = {
    "kb_qa":       KB_QA_CONFIG,
    "helpdesk":    HELPDESK_CONFIG,
    "compliance":  COMPLIANCE_CONFIG,
}


def get_config(use_case: str) -> AgentConfig:
    """根据use_case名称获取对应配置，不存在则抛出ValueError。"""
    if use_case not in USE_CASE_REGISTRY:
        raise ValueError(
            f"未知的use_case: '{use_case}'。"
            f"支持的场景: {list(USE_CASE_REGISTRY.keys())}"
        )
    return USE_CASE_REGISTRY[use_case]
