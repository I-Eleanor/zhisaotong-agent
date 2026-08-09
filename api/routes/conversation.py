"""对话接口（SSE 流式）。"""
from fastapi import APIRouter
from api.schemas import ChatRequest
from api.streaming import sse_bridge
from agent.orchestrator import get_orchestrator

router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """对话 Agent 流式输出，支持多轮记忆。

    返回 SSE 流，每个事件形如：event: message\\ndata: <AgentEvent JSON>
    """
    orchestrator = get_orchestrator()

    def run():
        return orchestrator.execute(request.query, request.history, request.mode)

    return sse_bridge(run)
