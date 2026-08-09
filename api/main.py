"""FastAPI 应用入口。

启动：uvicorn api.main:app --host 0.0.0.0 --port 8000
API 文档：http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import conversation, diagnostic, knowledge
from utils.config_handler import rag_conf, chroma_conf

app = FastAPI(title="智扫通 Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversation.router, prefix="/api")
app.include_router(diagnostic.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")


@app.get("/api/health")
async def health():
    """健康检查：返回服务状态、当前配置的模型与 embedding 路径。"""
    return {
        "status": "ok",
        "model": rag_conf.get("chat_model_name", "unknown"),
        "embedding": rag_conf.get("embedding_local_path", "unknown"),
        "reranker_enabled": chroma_conf.get("reranker_enabled", False),
    }
