"""API 请求/响应模型（Pydantic）。"""
from typing import Optional, List
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """对话请求。history 为多轮记忆，mode 可强制 routing。"""
    query: str
    history: Optional[List[dict]] = None
    mode: Optional[str] = None  # "conversation" | "diagnostic"


class DiagnoseRequest(BaseModel):
    """诊断请求。"""
    query: str


class KnowledgeUploadResponse(BaseModel):
    success: bool
    file_count: int


class KnowledgeRebuildResponse(BaseModel):
    success: bool
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding: str
