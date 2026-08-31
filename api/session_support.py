"""API 会话辅助：user_id 归一化、会话归属校验与安全 403 响应。

供 /api/chat 与 /api/diagnose 路由复用；不在 API 层触碰任何全局单例
（会话存储经 AppContainer 注入）。
"""
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from agent.services.device_registry import normalize_user_id
from agent.services.session_store import ConversationAccessDeniedError
from utils import error_codes
from utils.logger_handler import log_safe_text, logger
from utils.request_context import get_request_id, set_request_user_id, set_session_id

if TYPE_CHECKING:
    from api.container import AppContainer

DENIED_SAFE_MESSAGE = "无权访问该会话，请使用自己的会话标识。"


def denied_response(request_id: str = "") -> JSONResponse:
    """会话归属校验失败的统一 403 安全响应（三字段结构，不确认会话是否存在）。"""
    return JSONResponse(status_code=403, content={
        "error_code": error_codes.CONVERSATION_ACCESS_DENIED,
        "safe_message": DENIED_SAFE_MESSAGE,
        "request_id": request_id or get_request_id(),
    })


def resolve_session(container: "AppContainer", user_id: str | None, conversation_id: str | None) -> dict:
    """解析当前请求的会话并把 user_id / conversation_id 写入请求上下文。

    - user_id 未传时回退默认演示用户（1001），不再随机生成；
    - conversation_id 未传 / 不存在 → 服务端新建会话；
    - conversation_id 归属其他用户 → 抛 ConversationAccessDeniedError（路由层转 403）。
    """
    resolved_user_id = normalize_user_id(user_id)
    try:
        session = container.session_store.resolve(resolved_user_id, conversation_id)
    except ConversationAccessDeniedError:
        logger.warning({
            "event": "api_session_denied",
            "user_id": log_safe_text(resolved_user_id),
            "request_id": get_request_id(),
            "error_code": error_codes.CONVERSATION_ACCESS_DENIED,
        })
        raise

    set_request_user_id(resolved_user_id)
    set_session_id(session["conversation_id"])
    logger.info({
        "event": "api_session_resolved",
        "user_id": resolved_user_id,
        "conversation_id": session["conversation_id"],
        "history_messages": len(session["messages"]),
        "request_id": get_request_id(),
    })
    return session


__all__ = ["resolve_session", "denied_response", "DENIED_SAFE_MESSAGE"]
