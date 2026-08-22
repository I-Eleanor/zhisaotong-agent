"""对话接口（SSE 流式 + 同步兜底）。"""
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.container import AppContainer, get_app_container
from api.schemas import ChatRequest
from api.streaming import sse_bridge
from utils.exceptions import AgentProjectError, normalize_error_code
from utils.logger_handler import log_safe_text, logger
from utils.request_context import get_request_id

router = APIRouter()

# /chat/sync 错误事件的安全提示：固定文案，不透传 Agent 事件内容
# （事件 content 无法在路由层验证是否含敏感文本，只能进日志）
CHAT_SYNC_ERROR_SAFE_MESSAGE = "对话处理失败，请稍后重试。"


@router.post("/chat")
async def chat(request: ChatRequest, container: AppContainer = Depends(get_app_container)):  # noqa: B008
    """对话 Agent 流式输出，支持多轮记忆。

    返回 SSE 流，每个事件形如：event: message\\ndata: <AgentEvent JSON>
    """
    rid = get_request_id()
    orchestrator = container.orchestrator

    def run():
        return orchestrator.execute(request.query, request.history, request.mode)

    return sse_bridge(run, request_id=rid)


@router.post("/chat/sync")
def chat_sync(request: ChatRequest, container: AppContainer = Depends(get_app_container)):  # noqa: B008
    """同步对话接口（流式失败时的兜底）。一次性返回完整回答。

    普通 def：FastAPI 自动放入线程池执行，内部的同步 LLM 调用不会阻塞事件循环。
    错误响应使用统一结构 {error_code, safe_message, request_id}：
    - Agent error 事件 → 500 + 事件 error_code（未知码回退 INTERNAL_ERROR）；
    - 未预期异常 → 包装为 AgentProjectError 交全局 handler 统一转换。
    """
    rid = get_request_id()
    orchestrator = container.orchestrator

    try:
        answer = ""
        for event in orchestrator.execute(request.query, request.history, request.mode):
            etype = event.get("type", "")
            if etype == "message":
                answer += event.get("content", "")
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
        return {"answer": answer.strip()}
    except AgentProjectError:
        raise
    except Exception as e:
        raise AgentProjectError("对话处理失败", stage="sync_chat", original=e) from e
