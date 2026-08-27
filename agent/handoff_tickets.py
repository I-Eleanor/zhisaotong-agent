"""转人工工单轻量服务：供 Orchestrator 在用户显式转人工时创建工单。

职责边界：
- 仅在 HANDOFF_DB_PATH 已配置时提供 create()（配置缺失返回 None，由调用方
  决定降级行为，不在此抛异常）；
- 每次调用实时读取环境变量，便于测试时用 monkeypatch 切换路径；
- 自身不做任何凭证逻辑（access_key 的生成/哈希/剥离由 HandoffStore 负责）；
  Orchestrator 调用 create() 后丢弃明文 access_key——SSE 事件流不回传凭证。
"""
from typing import Any


class HandoffTicketService:
    """代理 HandoffStore 的轻量服务，返回 (ticket, access_key) 或 None。"""

    def create(
        self,
        user_question: str,
        conversation_id: str | None = None,
        recent_conversations: Any = None,
        conversation_summary: str = "",
        retrieval_sources: Any = None,
        diagnostic_result: str = "",
        handoff_reason: str = "user_request",
    ) -> tuple[dict, str] | None:
        """创建工单；HANDOFF_DB_PATH 未配置时返回 None（表示工单不可用）。"""
        import os

        if not os.getenv("HANDOFF_DB_PATH", "").strip():
            return None
        from api.handoff_store import HandoffStore

        store = HandoffStore()
        return store.create(
            user_question=user_question,
            conversation_id=conversation_id,
            recent_conversations=recent_conversations,
            conversation_summary=conversation_summary,
            retrieval_sources=retrieval_sources,
            diagnostic_result=diagnostic_result,
            handoff_reason=handoff_reason,
        )

    def get_active_by_conversation(self, conversation_id: str | None) -> dict | None:
        """某会话最近一张未结束（pending/processing）的工单；无则 None。"""
        if not conversation_id:
            return None
        import os

        from api.handoff_store import HandoffStore

        if not os.getenv("HANDOFF_DB_PATH", "").strip():
            return None
        return HandoffStore().get_active_by_conversation(conversation_id)

    def append_user_message(self, conversation_id: str | None, content: str) -> dict | None:
        """人工接管期间把用户消息追加到活动工单的实时对话日志；无则 None。"""
        if not conversation_id or not (content or "").strip():
            return None
        import os

        from api.handoff_store import HandoffStore

        if not os.getenv("HANDOFF_DB_PATH", "").strip():
            return None
        return HandoffStore().append_user_message(conversation_id, content.strip())
