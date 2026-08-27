"""API 请求/响应模型（Pydantic）。"""
from typing import Literal

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    """对话请求。history 为多轮记忆，mode 可强制 routing，
    conversation_id 为前端会话内稳定的会话标识（转人工关联也用）。"""
    query: str
    history: list[dict] | None = None
    mode: Literal["conversation", "diagnostic"] | None = None
    conversation_id: str | None = None


class DiagnoseRequest(BaseModel):
    """诊断请求。"""
    query: str


class HandoffCreateRequest(BaseModel):
    """创建人工转接工单请求。

    conversation_id 仅用于会话关联展示，不作为访问凭证；
    访问凭证 access_key 由服务端生成并在创建响应中返回一次。
    """
    user_question: str
    conversation_id: str | None = None
    recent_conversations: list | str | None = None
    conversation_summary: str = ""
    retrieval_sources: list | str | None = None
    diagnostic_result: str = ""
    handoff_reason: str = "user_request"

    @field_validator("user_question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_question 不能为空")
        return v


class AdminLoginRequest(BaseModel):
    """管理端登录请求。提交 ADMIN_TOKEN 换取服务端会话。"""
    admin_token: str


class HandoffReplyRequest(BaseModel):
    """管理端回复工单请求。status 缺省时置为 resolved。"""
    human_reply: str
    status: Literal["pending", "processing", "resolved", "closed"] | None = None


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
