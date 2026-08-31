"""对话接口（SSE 流式 + 同步兜底）。

会话模型（轻量演示级）：
- 请求可携带 user_id / conversation_id；未携带 user_id 时使用默认演示用户 1001；
- 未携带 conversation_id 时服务端新建会话，并在 SSE 首个 session 事件
  （或 /chat/sync 响应字段）中返回，客户端后续携带同一 ID 即可延续上下文；
- conversation_id 归属其他用户时返回 403 安全错误（不确认会话是否存在）；
- 携带 conversation_id 时历史以服务端会话为准（request.history 被忽略）；
  未携带时沿用 request.history（旧客户端格式完全兼容）。
"""
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from agent.events import EVENT_TYPE_SESSION, make_event
from agent.services.session_store import ConversationAccessDeniedError
from api.container import AppContainer, get_app_container
from api.schemas import ChatRequest
from api.session_support import denied_response, resolve_session
from api.streaming import sse_bridge
from utils.exceptions import AgentProjectError, normalize_error_code
from utils.logger_handler import log_safe_text, logger
from utils.request_context import get_request_id

router = APIRouter()

# /chat/sync 错误事件的安全提示：固定文案，不透传 Agent 事件内容
# （事件 content 无法在路由层验证是否含敏感文本，只能进日志）
CHAT_SYNC_ERROR_SAFE_MESSAGE = "对话处理失败，请稍后重试。"


def _session_event(conversation_id: str, user_id: str):
    """SSE 首个事件：携带会话标识，供前端保存并在后续请求中回传。"""
    return make_event(
        EVENT_TYPE_SESSION,
        agent="orchestrator",
        content="",
        conversation_id=conversation_id,
        user_id=user_id,
    )


@router.post("/chat")
async def chat(request: ChatRequest, container: AppContainer = Depends(get_app_container)):  # noqa: B008
    """对话 Agent 流式输出，支持多轮记忆（服务端会话或客户端 history）。

    返回 SSE 流，首个事件为 session（conversation_id / user_id），
    之后每个事件形如：event: message\\ndata: <AgentEvent JSON>
    """
    rid = get_request_id()
    orchestrator = container.orchestrator

    try:
        session = resolve_session(container, request.user_id, request.conversation_id)
    except ConversationAccessDeniedError:
        return denied_response(rid)

    conversation_id = session["conversation_id"]
    user_id = session["user_id"]
    # 携带 conversation_id 的请求以服务端会话历史为准；旧格式（无会话 ID）
    # 继续使用客户端传入的 history，行为完全兼容
    history = session["messages"] if request.conversation_id else (request.history or [])
    store = container.session_store

    def run():
        def gen():
            yield _session_event(conversation_id, user_id)
            answer_parts: list[str] = []
            try:
                for event in orchestrator.execute(
                    request.query, history, request.mode,
                    conversation_id=conversation_id,
                ):
                    yield event
                    if event.get("type") == "message":
                        answer_parts.append(event.get("content", ""))
            finally:
                # 会话落库：用户提问必存；助手回答（可能因客户端断开而是部分内容）尽力保存
                store.append_message(conversation_id, "user", request.query)
                answer = "".join(answer_parts).strip()
                if answer:
                    store.append_message(conversation_id, "assistant", answer)

        return gen()

    return sse_bridge(run, request_id=rid)


@router.post("/chat/sync")
def chat_sync(request: ChatRequest, container: AppContainer = Depends(get_app_container)):  # noqa: B008
    """同步对话接口（流式失败时的兜底）。一次性返回完整回答。

    普通 def：FastAPI 自动放入线程池执行，内部的同步 LLM 调用不会阻塞事件循环。
    错误响应使用统一结构 {error_code, safe_message, request_id}：
    - Agent error 事件 → 500 + 事件 error_code（未知码回退 INTERNAL_ERROR）；
    - 未预期异常 → 包装为 AgentProjectError 交全局 handler 统一转换。
    响应额外携带 conversation_id / user_id（旧客户端可忽略）。
    """
    rid = get_request_id()
    orchestrator = container.orchestrator

    try:
        session = resolve_session(container, request.user_id, request.conversation_id)
    except ConversationAccessDeniedError:
        return denied_response(rid)

    conversation_id = session["conversation_id"]
    user_id = session["user_id"]
    history = session["messages"] if request.conversation_id else (request.history or [])
    store = container.session_store

    try:
        answer = ""
        need_handoff = False
        store.append_message(conversation_id, "user", request.query)
        for event in orchestrator.execute(
            request.query, history, request.mode,
            conversation_id=conversation_id,
        ):
            etype = event.get("type", "")
            if etype == "message":
                answer += event.get("content", "")
            elif etype == "handoff_suggested" or etype == "handoff_created":
                # 转人工事件：不计入 answer，只置标记（工单由前端/调用方提交 POST /api/handoff）
                need_handoff = True
            elif etype == "error":
                data = event.get("data") or {}
                code = normalize_error_code(data.get("error_code"))
                logger.warning({
                    "event": "chat_sync_agent_error",
                    "request_id": rid,
                    "error_code": code,
                    "agent_error_content": log_safe_text(event.get("content", "")),
                    "stage": "sync_chat",
                })
                return JSONResponse(status_code=500, content={
                    "error_code": code,
                    "safe_message": CHAT_SYNC_ERROR_SAFE_MESSAGE,
                    "request_id": rid,
                })
        final_answer = answer.strip()
        if final_answer:
            store.append_message(conversation_id, "assistant", final_answer)
        return {
            "answer": final_answer,
            "need_handoff": need_handoff,
            "conversation_id": conversation_id,
            "user_id": user_id,
        }
    except AgentProjectError:
        raise
    except Exception as e:
        raise AgentProjectError("对话处理失败", stage="sync_chat", original=e) from e
