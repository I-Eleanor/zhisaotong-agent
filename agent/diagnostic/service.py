"""诊断 Agent 对外服务（DiagnosticAgent）。

事件类型（与 agent/events.py 协议保持兼容）：plan / step / replan / report / error / done。
"""
import threading
from collections.abc import Iterator

from agent.diagnostic.graph import build_diagnostic_graph
from agent.diagnostic.parser import LlmParser
from agent.diagnostic.tool_router import ToolRouter
from agent.events import AgentEvent, make_event
from utils import error_codes
from utils.logger_handler import log_safe_text, logger, safe_exception_fields
from utils.request_context import get_request_id


class DiagnosticAgent:
    """Plan-Execute-Replan 诊断 Agent；parser / tool_router / model 可注入便于测试。"""

    def __init__(self, parser: LlmParser | None = None, tool_router: ToolRouter | None = None, model=None):
        self.graph = build_diagnostic_graph(parser=parser, tool_router=tool_router, model=model)

    def run(self, user_query: str) -> Iterator[AgentEvent]:
        """流式执行诊断流程，逐个产出 plan/step/replan/report 事件，最后必发 done。"""
        from agent.diagnostic.nodes import initial_state

        initial = initial_state(user_query)
        seen = 0
        try:
            for snapshot in self.graph.stream(initial, stream_mode="values"):
                evs = snapshot.get("events", []) or []
                yield from evs[seen:]
                seen = len(evs)
        except Exception as e:
            logger.error({
                "event": "diagnostic_agent_error",
                "query": log_safe_text(user_query),
                "request_id": get_request_id(),
                "stage": "diagnostic",
                **safe_exception_fields(e),
            })
            # 客户端只收到安全提示与结构化错误字段，原始异常仅进日志
            yield make_event(
                "error",
                agent="diagnostic",
                content="诊断流程执行失败，请稍后重试。",
                error_code=error_codes.INTERNAL_ERROR,
                request_id=get_request_id(),
            )
        yield make_event("done", agent="diagnostic")


_diagnostic_agent: DiagnosticAgent | None = None
_agent_lock = threading.Lock()


def get_diagnostic_agent() -> DiagnosticAgent:
    """全局懒加载单例（图结构只编译一次；双检锁防并发重复初始化）。"""
    global _diagnostic_agent
    if _diagnostic_agent is None:
        with _agent_lock:
            if _diagnostic_agent is None:
                _diagnostic_agent = DiagnosticAgent()
    return _diagnostic_agent


def reset_diagnostic_agent() -> None:
    """重置单例，供测试隔离使用。"""
    global _diagnostic_agent
    _diagnostic_agent = None
