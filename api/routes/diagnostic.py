"""诊断接口（SSE 流式）。"""
from fastapi import APIRouter

from agent.orchestrator import get_orchestrator
from api.schemas import DiagnoseRequest
from api.streaming import sse_bridge
from utils.request_context import get_request_id

router = APIRouter()


@router.post("/diagnose")
async def diagnose(request: DiagnoseRequest):
    """诊断 Agent 流式输出排查过程 + 最终报告。

    返回 SSE 流，事件 type 含 plan / step / replan / report。
    """
    rid = get_request_id()
    orchestrator = get_orchestrator()

    def run():
        return orchestrator.execute(request.query, mode="diagnostic")

    return sse_bridge(run, request_id=rid)
