"""FastAPI entry point: REST endpoints and agent lifecycle management. Supports regular
responses, SSE streaming, health checks, and metrics."""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.factory import AgentFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
# Shared agent registry warmed up at startup.
# ─────────────────────────────────────────────────────
_agents: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service startup and shutdown."""
    logger.info("Initializing agents for all use cases...")
    try:
        _agents.update(AgentFactory.create_all())
        logger.info(f"Agents ready: {list(_agents.keys())}")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        # Alert on warmup failures in production; this example continues startup.

    yield  # Serve requests.

    # Release resources at shutdown.
    from agent.factory import get_shared_cache
    try:
        await get_shared_cache().close()
    except Exception:
        pass
    logger.info("Service shutdown complete.")


# ─────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG Agent API",
    description="生产级多场景LLM Agent，支持kb_qa / helpdesk / compliance",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# Request and response models.
# ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="用户查询")
    use_case: str = Field(..., description="业务场景: kb_qa | helpdesk | compliance")
    tenant_id: str = Field(..., description="企业租户ID，用于数据隔离")
    stream: bool = Field(False, description="是否流式输出")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "员工年假政策是什么？",
                "use_case": "kb_qa",
                "tenant_id": "acme_corp",
                "stream": False,
            }
        }


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    latency_ms: int
    use_case: str
    from_cache: bool


class HealthResponse(BaseModel):
    status: str
    agents_ready: list[str]
    uptime_seconds: float


# ─────────────────────────────────────────────────────
# Start time for uptime calculation.
# ─────────────────────────────────────────────────────
_start_time = time.time()


# ─────────────────────────────────────────────────────
# API endpoints.
# ─────────────────────────────────────────────────────

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming chat endpoint. Route requests by use case and run caching, retrieval,
    and generation. Target P50 latency: <= 2000 ms."""
    if req.use_case not in _agents:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的use_case: {req.use_case}。"
                   f"可用场景: {list(_agents.keys())}"
        )

    agent = _agents[req.use_case]
    try:
        result = await agent.run(
            query=req.query,
            tenant_id=req.tenant_id,
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Agent error | use_case={req.use_case} | error={e}")
        raise HTTPException(status_code=500, detail=f"Agent执行失败: {str(e)}")


@app.post("/v1/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming chat endpoint for interactive applications. Target time to first
    token: < 500 ms."""
    if req.use_case not in _agents:
        raise HTTPException(status_code=400, detail=f"不支持的use_case: {req.use_case}")

    agent = _agents[req.use_case]

    async def event_generator():
        try:
            async for token in agent.run_stream(req.query, req.tenant_id):
                # SSE data frame followed by two newlines.
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx SSE buffering.
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for a Kubernetes liveness probe."""
    return HealthResponse(
        status="healthy",
        agents_ready=list(_agents.keys()),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics")
async def metrics():
    """Expose latency metrics. Consider prometheus_fastapi_instrumentator for production
    monitoring."""
    from agent.factory import get_shared_llm_client

    llm_stats = get_shared_llm_client().get_latency_stats()

    pipeline_stats = {}
    for use_case, agent in _agents.items():
        pipeline_stats[use_case] = agent.retriever.get_latency_stats()

    from agent.factory import get_shared_cache
    cache_stats = get_shared_cache().get_stats()

    return {
        "llm_latency": llm_stats,
        "retrieval_latency": pipeline_stats,
        "cache": cache_stats,
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.delete("/cache/{use_case}/{tenant_id}")
async def invalidate_cache(use_case: str, tenant_id: str):
    """Invalidate a tenant cache after document updates."""
    from agent.factory import get_shared_cache
    deleted = await get_shared_cache().invalidate(use_case, tenant_id)
    return {"deleted_keys": deleted}


# ─────────────────────────────────────────────────────
# Application entry point.
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,             # Worker processes.
        loop="uvloop",         # Use uvloop.
        log_level="info",
    )
