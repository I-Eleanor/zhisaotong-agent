"""Agent 统一事件协议。

三个 Agent 与编排层对外一律以「事件字典」的形式流式输出，
上层（React 前端 / FastAPI SSE）只需要按 type 分支渲染，无需感知 Agent 内部实现。

事件类型：
    route       编排层完成意图路由
    thinking    Agent 调用工具前的思考过程（浅色显示，最终回答前消失）
    message     Agent 产出的最终自然语言回答
    tool_start  工具开始调用
    tool_end    工具调用结束
    plan        诊断 Agent 生成/更新排查计划
    step        诊断 Agent 完成一个排查步骤
    replan      诊断 Agent 重规划决策
    report      诊断 Agent 产出最终诊断报告
    error       执行过程中发生异常
    done        本次执行结束
"""
from collections.abc import Iterator
from typing import Any, TypedDict


class AgentEvent(TypedDict, total=False):
    type: str
    agent: str
    content: str
    data: dict[str, Any]


EVENT_TYPE_ROUTE: str = "route"
EVENT_TYPE_THINKING: str = "thinking"
EVENT_TYPE_MESSAGE: str = "message"
EVENT_TYPE_TOOL_START: str = "tool_start"
EVENT_TYPE_TOOL_END: str = "tool_end"
EVENT_TYPE_PLAN: str = "plan"
EVENT_TYPE_STEP: str = "step"
EVENT_TYPE_REPLAN: str = "replan"
EVENT_TYPE_REPORT: str = "report"
EVENT_TYPE_ERROR: str = "error"
EVENT_TYPE_DONE: str = "done"

VALID_EVENT_TYPES: frozenset[str] = frozenset({
    EVENT_TYPE_ROUTE, EVENT_TYPE_THINKING, EVENT_TYPE_MESSAGE, EVENT_TYPE_TOOL_START, EVENT_TYPE_TOOL_END,
    EVENT_TYPE_PLAN, EVENT_TYPE_STEP, EVENT_TYPE_REPLAN, EVENT_TYPE_REPORT,
    EVENT_TYPE_ERROR, EVENT_TYPE_DONE,
})


def make_event(event_type: str, agent: str = "", content: str = "", **data: Any) -> AgentEvent:
    event: AgentEvent = {"type": event_type, "agent": agent, "content": content}
    if data:
        event["data"] = data
    return event


def event_to_text(event: AgentEvent) -> str:
    """把事件降级为纯文本，便于日志、CLI 以及不关心结构的消费方使用。"""
    event_type = event.get("type", "")
    content = event.get("content", "") or ""
    data = event.get("data", {}) or {}

    if event_type in (EVENT_TYPE_MESSAGE, EVENT_TYPE_REPORT):
        return content

    if event_type == EVENT_TYPE_THINKING:
        return f"[思考] {content}\n"

    if event_type == EVENT_TYPE_ROUTE:
        return f"[路由] 本次请求交由「{data.get('mode_label', content)}」处理\n"

    if event_type == EVENT_TYPE_PLAN:
        steps = data.get("steps", [])
        lines = "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, start=1))
        return f"[排查计划]\n{lines}\n"

    if event_type == EVENT_TYPE_STEP:
        return f"[步骤{data.get('index', '?')}] {data.get('description', '')}\n{content}\n"

    if event_type == EVENT_TYPE_REPLAN:
        return f"[重规划] {content}\n"

    if event_type == EVENT_TYPE_TOOL_START:
        return f"[调用工具] {data.get('tool', '')}\n"

    if event_type == EVENT_TYPE_ERROR:
        return f"[错误] {content}\n"

    return content


def events_to_text(events: Iterator[AgentEvent]) -> Iterator[str]:
    """把事件流转换为纯文本流，过滤掉空片段。"""
    for event in events:
        text = event_to_text(event)
        if text:
            yield text
