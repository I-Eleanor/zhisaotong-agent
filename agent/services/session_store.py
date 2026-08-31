"""轻量对话会话存储（演示级，进程内内存实现）。

会话模型：
    conversation_id（服务端生成）
      → user_id（会话归属：其他用户携带该 ID 一律拒绝）
      → messages（[{role: user|assistant, content: str}]，仅上下文用途）
      → created_at / updated_at

设计边界（明确不做）：
- 不实现用户注册 / 登录 / OAuth / 复杂权限系统；
- 不保存真实设备状态——设备数据始终由 MCP 实时查询，会话只保存对话上下文；
- 进程内存存储，服务重启后会话丢失（客户端此时会拿到新的 conversation_id）。

会话 ID 一律由服务端生成（防抢占）；客户端只透传。归属校验失败
抛 ConversationAccessDeniedError，API 层转换为固定安全错误响应。
"""
import threading
import uuid
from datetime import UTC, datetime

from utils import error_codes
from utils.exceptions import AgentProjectError
from utils.logger_handler import log_safe_text, logger

# 单会话保存的最大消息条数（user + assistant 合计；超出后丢弃最早的消息）
MAX_SESSION_MESSAGES = 50


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ConversationAccessDeniedError(AgentProjectError):
    """会话归属校验失败：请求携带的 conversation_id 不属于当前 user_id。

    message 只进服务端日志；safe_message 为固定文案，不确认会话是否存在。
    """

    stage = "session"
    retryable = False
    error_code = error_codes.CONVERSATION_ACCESS_DENIED
    safe_message = "无权访问该会话，请使用自己的会话标识。"


class ConversationSessionStore:
    """线程安全的内存会话存储。"""

    def __init__(self, max_messages: int = MAX_SESSION_MESSAGES):
        self._lock = threading.Lock()
        self._sessions: dict[str, dict] = {}
        self.max_messages = max_messages

    # --------------------------------------------------------------- 创建 / 解析

    def create(self, user_id: str) -> dict:
        """为指定用户创建新会话，返回会话快照。"""
        conversation_id = f"conv-{uuid.uuid4().hex[:12]}"
        now = _now_iso()
        session = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "messages": [],
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._sessions[conversation_id] = session
        logger.info({
            "event": "session_created",
            "conversation_id": conversation_id,
            "user_id": log_safe_text(user_id),
        })
        return self.snapshot(session)

    def resolve(self, user_id: str, conversation_id: str | None) -> dict:
        """解析（或创建）当前请求所属的会话。

        规则：
        - 未携带 conversation_id → 创建新会话；
        - 携带但不存在（如服务重启后内存丢失）→ 创建新会话（新 ID），
          客户端从响应中更新 conversation_id；
        - 携带且存在但归属其他用户 → 抛 ConversationAccessDeniedError。
        """
        if not conversation_id:
            return self.create(user_id)

        with self._lock:
            session = self._sessions.get(conversation_id)

        if session is None:
            logger.info({
                "event": "session_not_found_new_created",
                "conversation_id": log_safe_text(conversation_id),
            })
            return self.create(user_id)

        if session.get("user_id") != user_id:
            logger.warning({
                "event": "session_access_denied",
                "conversation_id": log_safe_text(conversation_id),
                "user_id": log_safe_text(user_id),
                "error_code": error_codes.CONVERSATION_ACCESS_DENIED,
            })
            raise ConversationAccessDeniedError("会话归属校验失败")

        logger.info({
            "event": "session_resumed",
            "conversation_id": conversation_id,
            "messages": len(session.get("messages", [])),
        })
        return self.snapshot(session)

    # --------------------------------------------------------------- 读写消息

    def append_message(self, conversation_id: str, role: str, content: str) -> None:
        """追加一条消息（user / assistant）；超出上限时丢弃最早的消息。

        空白内容不写入（与 ConversationBuffer 一致），避免把空回答
        或纯空白输入污染上下文。
        """
        text = (content or "").strip()
        if not conversation_id or not text or role not in ("user", "assistant"):
            return
        with self._lock:
            session = self._sessions.get(conversation_id)
            if session is None:
                return
            session["messages"].append({"role": role, "content": text})
            if len(session["messages"]) > self.max_messages:
                session["messages"] = session["messages"][-self.max_messages:]
            session["updated_at"] = _now_iso()

    def history(self, conversation_id: str) -> list[dict]:
        """返回会话历史消息副本（无会话时为空列表）。"""
        with self._lock:
            session = self._sessions.get(conversation_id)
            return list(session["messages"]) if session else []

    def get(self, conversation_id: str) -> dict | None:
        """返回会话快照；不存在返回 None。"""
        with self._lock:
            session = self._sessions.get(conversation_id)
            return self.snapshot(session) if session else None

    def size(self) -> int:
        """当前会话总数（监控/测试用）。"""
        with self._lock:
            return len(self._sessions)

    # --------------------------------------------------------------- 内部

    @staticmethod
    def snapshot(session: dict) -> dict:
        """返回会话的浅拷贝快照（messages 为新列表，防外部篡改）。"""
        return {
            "conversation_id": session["conversation_id"],
            "user_id": session["user_id"],
            "messages": list(session["messages"]),
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
        }


__all__ = [
    "ConversationSessionStore",
    "ConversationAccessDeniedError",
    "MAX_SESSION_MESSAGES",
]
