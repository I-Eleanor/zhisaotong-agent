"""诊断接口（SSE 流式）。

会话模型与 /api/chat 一致：
- user_id 未传时使用默认演示用户 1001；
- conversation_id 未传时服务端新建会话，SSE 首个 session 事件返回；
- conversation_id 归属其他用户时返回 403 安全错误；
- 同一会话的历史（此前的诊断提问与报告）作为本轮诊断的上下文，
  设备状态 / 日志数据仍由 MCP 工具实时查询（模拟设备数据）。
"""
from fastapi import APIRouter, Depends

from agent.events import EVENT_TYPE_SESSION, make_event
from agent.services.session_store import ConversationAccessDeniedError
from api.container import AppContainer, get_app_container
from api.schemas import DiagnoseRequest
from api.session_support import denied_response, resolve_session
from api.streaming import sse_bridge
from utils.request_context import get_request_id

router = APIRouter()


@router.post("/diagnose")
async def diagnose(request: DiagnoseRequest, container: AppContainer = Depends(get_app_container)):  # noqa: B008
    """诊断 Agent 流式输出排查过程 + 最终报告。

    返回 SSE 流，首个事件为 session（conversation_id / user_id），
    之后事件 type 含 plan / step / replan / report / done。
    """
    rid = get_request_id()
    orchestrator = container.orchestrator

    try:
        session = resolve_session(container, request.user_id, request.conversation_id)
    except ConversationAccessDeniedError:
        return denied_response(rid)

    conversation_id = session["conversation_id"]
    user_id = session["user_id"]
    history = session["messages"]
    store = container.session_store

    def run():
        def gen():
            yield make_event(
                EVENT_TYPE_SESSION,
                agent="orchestrator",
                content="",
                conversation_id=conversation_id,
                user_id=user_id,
            )
            report_parts: list[str] = []
            try:
                for event in orchestrator.execute(
                    request.query, history, mode="diagnostic",
                    conversation_id=conversation_id,
                ):
                    yield event
                    if event.get("type") == "report":
                        report_parts.append(event.get("content", ""))
            finally:
                # 会话落库：本轮故障描述 + 最终诊断报告（供后续轮次读取上下文）
                store.append_message(conversation_id, "user", request.query)
                report = "".join(report_parts).strip()
                if report:
                    store.append_message(conversation_id, "assistant", report)

        return gen()

    return sse_bridge(run, request_id=rid)
