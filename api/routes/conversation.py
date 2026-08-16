"""对话接口（SSE 流式 + 同步兜底）。"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from agent.orchestrator import get_orchestrator
from api.schemas import ChatRequest
from api.streaming import sse_bridge
from utils.logger_handler import logger
from utils.request_context import get_request_id

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """对话 Agent 流式输出，支持多轮记忆。

    返回 SSE 流，每个事件形如：event: message\\ndata: <AgentEvent JSON>
    """
    rid = get_request_id()
    orchestrator = get_orchestrator()

    def run():
        return orchestrator.execute(request.query, request.history, request.mode)

    return sse_bridge(run, request_id=rid)


@router.post("/chat/sync")
async def chat_sync(request: ChatRequest):
    """同步对话接口（流式失败时的兜底）。一次性返回完整回答。"""
    rid = get_request_id()
    orchestrator = get_orchestrator()

    try:
        answer = ""
        for event in orchestrator.execute(request.query, request.history, request.mode):
            etype = event.get("type", "")
            if etype == "message":
                answer += event.get("content", "")
            elif etype == "error":
                return JSONResponse(
                    status_code=500,
                    content={"error": event.get("content", "处理失败")},
                )
        return {"answer": answer.strip()}
    except Exception as e:
        logger.error({"event": "chat_sync_error", "request_id": rid, "error": str(e)})
        return JSONResponse(status_code=500, content={"error": str(e)})
