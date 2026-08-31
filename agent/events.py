"""Agent 统一事件协议。

三个 Agent 与编排层对外一律以「事件字典」的形式流式输出，
上层（React 前端 / FastAPI SSE）只需要按 type 分支渲染，无需感知 Agent 内部实现。

事件类型：
    session     会话信息（SSE 首个事件：conversation_id / user_id）
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
    handoff_suggested  建议转人工（用户显式请求等触发）
    handoff_created    已创建人工转接工单（预留，当前未产出）
"""
from collections.abc import Iterator
from typing import Any, TypedDict


class AgentEvent(TypedDict, total=False):
    type: str
    agent: str
    content: str
    data: dict[str, Any]


EVENT_TYPE_SESSION: str = "session"
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
EVENT_TYPE_HANDOFF_SUGGESTED: str = "handoff_suggested"
EVENT_TYPE_HANDOFF_CREATED: str = "handoff_created"
EVENT_TYPE_HANDOFF_HUMAN_SERVICE: str = "handoff_human_service"

VALID_EVENT_TYPES: frozenset[str] = frozenset({
    EVENT_TYPE_SESSION, EVENT_TYPE_ROUTE, EVENT_TYPE_THINKING, EVENT_TYPE_MESSAGE,
    EVENT_TYPE_TOOL_START, EVENT_TYPE_TOOL_END,
    EVENT_TYPE_PLAN, EVENT_TYPE_STEP, EVENT_TYPE_REPLAN, EVENT_TYPE_REPORT,
    EVENT_TYPE_ERROR, EVENT_TYPE_DONE,
    EVENT_TYPE_HANDOFF_SUGGESTED, EVENT_TYPE_HANDOFF_CREATED, EVENT_TYPE_HANDOFF_HUMAN_SERVICE,
})

# 关键业务事件：在 SSE 桥中不允许因有界队列满被丢弃，且必须保持生产顺序。
# 转人工事件族（suggested → created → human_service）必须按序透传；error / done 属于必达的
# 结束类事件，同样纳入——转人工流程全走无界控制队列，严格 FIFO，
# 不会因"业务队列优先排空"而被提前输出或丢失。
#
# session 事件（携带 conversation_id）刻意不在此列：控制队列与业务队列是两个通道，
# 消费端优先排空业务队列，若 session 走控制通道会被后到的业务事件抢先输出，
# 破坏"SSE 首个事件返回会话标识"的约定。它是生产线程产出的第一个事件，
# 走业务队列即可保证首达且不可能被丢弃（队列满只可能发生在大量事件之后）。
NON_DROPPABLE_EVENT_TYPES: frozenset[str] = frozenset({
    EVENT_TYPE_HANDOFF_SUGGESTED, EVENT_TYPE_HANDOFF_CREATED, EVENT_TYPE_HANDOFF_HUMAN_SERVICE,
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

    if event_type == EVENT_TYPE_SESSION:
        return f"[会话] {data.get('conversation_id', '')}\n"

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

    if event_type in (EVENT_TYPE_HANDOFF_SUGGESTED, EVENT_TYPE_HANDOFF_CREATED):
        return f"[转人工] {content}\n"

    if event_type == EVENT_TYPE_HANDOFF_HUMAN_SERVICE:
        return f"[人工服务] {content}\n"

    return content


def events_to_text(events: Iterator[AgentEvent]) -> Iterator[str]:
    """把事件流转换为纯文本流，过滤掉空片段。"""
    for event in events:
        text = event_to_text(event)
        if text:
            yield text
