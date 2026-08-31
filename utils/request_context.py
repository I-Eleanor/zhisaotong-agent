"""请求级上下文：request_id 贯穿 FastAPI → Orchestrator → Agent → Tool → RAG → 日志。

使用 contextvars 在异步/线程环境中传递 request_id。
"""
import uuid
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def get_request_id() -> str:
    rid = request_id_var.get()
    if not rid:
        rid = new_request_id()
        request_id_var.set(rid)
    return rid


def set_request_id(rid: str) -> None:
    request_id_var.set(rid)


def get_session_id() -> str:
    return session_id_var.get()


def set_session_id(sid: str) -> None:
    session_id_var.set(sid)


def get_request_user_id() -> str:
    """当前请求的用户 ID（API 层写入；未设置时为空字符串）。"""
    return user_id_var.get()


def set_request_user_id(user_id: str) -> None:
    user_id_var.set(user_id)
