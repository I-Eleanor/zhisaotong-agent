"""诊断接口（SSE 流式）。"""
from fastapi import APIRouter, Depends

from api.container import AppContainer, get_app_container
from api.schemas import DiagnoseRequest
from api.streaming import sse_bridge
from utils.request_context import get_request_id

router = APIRouter()


@router.post("/diagnose")
async def diagnose(request: DiagnoseRequest, container: AppContainer = Depends(get_app_container)):  # noqa: B008
    """诊断 Agent 流式输出排查过程 + 最终报告。

    返回 SSE 流，事件 type 含 plan / step / replan / report。
    """
    rid = get_request_id()
    orchestrator = container.orchestrator

    def run():
        return orchestrator.execute(request.query, mode="diagnostic")

    return sse_bridge(run, request_id=rid)
