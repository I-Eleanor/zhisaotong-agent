"""FastAPI 应用入口。

启动：uvicorn api.main:app --host 0.0.0.0 --port 8000
API 文档：http://localhost:8000/docs
"""
import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.routes import conversation, diagnostic, knowledge
from api.security import RequestSizeLimitMiddleware, TokenAuthMiddleware, limiter
from utils.config_handler import chroma_conf, rag_conf
from utils.logger_handler import logger
from utils.request_context import new_request_id, set_request_id

app = FastAPI(title="智扫通 Agent API", version="2.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in _cors_origins if o.strip()], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TokenAuthMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", "") or new_request_id()
    set_request_id(rid)
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    logger.info({
        "event": "request_completed",
        "request_id": rid,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": duration_ms,
    })
    return response


app.include_router(conversation.router, prefix="/api")
app.include_router(diagnostic.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")


@app.get("/api/health")
async def health():
    """健康检查：返回服务状态。"""
    return {
        "status": "ok",
        "model": rag_conf.get("chat_model_name", "unknown"),
        "reranker_enabled": chroma_conf.get("reranker_enabled", False),
    }


@app.get("/api/health/live")
async def health_live():
    """存活探针：进程是否存活。"""
    return {"status": "alive"}


@app.get("/api/health/ready")
async def health_ready():
    """就绪探针：配置、知识库和必要依赖是否可用。"""
    checks = {}

    try:
        from rag.vector_store import VectorStoreService
        vs = VectorStoreService()
        count = vs.vector_store._collection.count()
        checks["vector_store"] = {"ok": True, "chunk_count": count}
    except Exception:
        checks["vector_store"] = {"ok": False, "error": "向量库不可用"}

    try:
        from model.factory import get_chat_model
        get_chat_model()
        checks["chat_model"] = {"ok": True}
    except Exception:
        checks["chat_model"] = {"ok": False, "error": "模型初始化失败"}

    all_ok = all(c.get("ok", False) for c in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}
