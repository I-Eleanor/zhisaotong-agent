"""诊断 Agent 节点（Plan-Execute-Replan）。

状态机（不再用 plan + current_step_index 推导进度）：
    pending_steps    待执行步骤队列，executor 每次只消费第一项
    completed_steps  已完成步骤（步骤 + 结构化执行结果）
    iteration_count  已执行轮次，达到 MAX_ITERATIONS 强制结束

replanner 返回的步骤整体替换 pending_steps（新待执行步骤，从第一项开始），
并过滤与已完成步骤完全重复（tool/description/arguments 均相同）的项。
"""
from typing import Any, Protocol, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.diagnostic.parser import LlmParser
from agent.diagnostic.schemas import (
    MAX_ITERATIONS,
    MAX_STEPS,
    CompletedStep,
    DiagnosticPlan,
    DiagnosticStep,
)
from agent.diagnostic.tool_router import ToolRouter
from agent.events import AgentEvent, make_event
from model.factory import get_chat_model
from utils.logger_handler import log_safe_text, log_safe_value, logger, safe_exception_fields
from utils.prompt_loader import (
    load_diagnostic_plan_prompt,
    load_diagnostic_replan_prompt,
    load_diagnostic_report_prompt,
)

AGENT_NAME = "diagnostic"


class ChatModelLike(Protocol):
    """reporter 依赖的最小模型接口（真实模型与测试替身都满足）。"""

    def invoke(self, messages: list[BaseMessage], **kwargs: Any) -> Any: ...


class DiagnosticState(TypedDict):
    """诊断流程状态（LangGraph 状态模式：TypedDict，替换语义 channel）。"""

    user_query: str
    pending_steps: list[DiagnosticStep]
    completed_steps: list[CompletedStep]
    iteration_count: int
    final_report: str
    should_end: bool
    events: list[AgentEvent]


def initial_state(user_query: str) -> dict:
    """构造初始状态。"""
    return {
        "user_query": user_query,
        "pending_steps": [],
        "completed_steps": [],
        "iteration_count": 0,
        "final_report": "",
        "should_end": False,
        "events": [],
    }


def _fallback_plan(user_query: str) -> DiagnosticPlan:
    """LLM 计划不可用时的固定兜底计划，保证流程不中断。"""
    return DiagnosticPlan(steps=[
        DiagnosticStep(description="查询设备运行状态", tool="query_device_status", arguments={}),
        DiagnosticStep(description="检索故障排除相关资料", tool="retrieve_knowledge", arguments={}),
        DiagnosticStep(description="检索维护保养建议", tool="query_maintenance", arguments={}),
    ])


def _is_duplicate(step: DiagnosticStep, completed: list[CompletedStep]) -> bool:
    """与已完成步骤完全一致（tool/description/arguments）视为重复，不再执行。"""
    return any(
        c.step.tool == step.tool
        and c.step.description == step.description
        and c.step.arguments == step.arguments
        for c in completed
    )


def _dedupe_steps(steps: list[DiagnosticStep], completed: list[CompletedStep]) -> list[DiagnosticStep]:
    """稳定去重：按 (tool, description, arguments) 首见保留，过滤与已完成重复的项。"""
    seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()
    out: list[DiagnosticStep] = []
    for s in steps:
        key = (s.tool, s.description, tuple(sorted(s.arguments.items())))
        if key in seen or _is_duplicate(s, completed):
            continue
        seen.add(key)
        out.append(s)
    return out


# ----------------------------------------------------------- planner

def planner_node(state: dict, parser: LlmParser | None = None) -> dict:
    """根据故障描述生成结构化排查计划；LLM 不可用/输出非法时使用固定兜底计划。"""
    query = state["user_query"]
    p = parser if parser is not None else LlmParser()
    plan = p.parse_plan(
        load_diagnostic_plan_prompt(),
        f"用户故障描述：\n{query}\n\n请生成本次排查计划（JSON，最多 {MAX_STEPS} 个步骤）。",
    )
    if plan is None or not plan.steps:
        logger.warning({"event": "planner_fallback", "query": log_safe_text(query)})
        plan = _fallback_plan(query)

    steps = _dedupe_steps(plan.steps, [])[:MAX_STEPS]
    events = list(state.get("events", []))
    events.append(make_event("plan", agent=AGENT_NAME, content="", steps=[s.description for s in steps]))

    logger.info({"event": "planner_done", "steps": log_safe_value([s.description for s in steps])})
    return {
        "pending_steps": steps,
        "iteration_count": state.get("iteration_count", 0),
        "should_end": False,
        "events": events,
    }


# ----------------------------------------------------------- executor

def executor_node(state: dict, tool_router: ToolRouter | None = None) -> dict:
    """执行 pending_steps 的第一项；队列为空则直接进入报告。"""
    pending = list(state.get("pending_steps", []))
    completed = list(state.get("completed_steps", []))
    events = list(state.get("events", []))

    if not pending:
        return {"should_end": True, "events": events}

    step = pending[0]
    router = tool_router if tool_router is not None else ToolRouter()
    result = router.execute(step, user_query=state.get("user_query", ""))

    completed.append(CompletedStep(step=step, result=result))
    events.append(make_event(
        "step",
        agent=AGENT_NAME,
        content=result.content if result.success else result.safe_error_message,
        index=len(completed),
        description=step.description,
        tool=step.tool,
        error_code=result.error_code if not result.success else "",
    ))

    logger.info({
        "event": "executor_step",
        "step": log_safe_text(step.description),
        "tool": log_safe_text(step.tool),
        "success": result.success,
        "error_code": result.error_code,
    })
    return {
        "pending_steps": pending[1:],
        "completed_steps": completed,
        "events": events,
    }


# ----------------------------------------------------------- replanner

def _format_completed(completed: list[CompletedStep]) -> str:
    """把已完成步骤格式化为 replanner / reporter 可读的过程文本。"""
    lines = []
    for i, c in enumerate(completed, start=1):
        if c.result.success:
            lines.append(f"【步骤{i}：{c.step.description}（工具 {c.step.tool}）】\n{c.result.content}")
        else:
            lines.append(
                f"【步骤{i}：{c.step.description}（工具 {c.step.tool}）】\n"
                f"（工具不可用/调用失败：{c.result.safe_error_message}）"
            )
    return "\n".join(lines)


def replanner_node(state: dict, parser: LlmParser | None = None) -> dict:
    """根据执行结果决策 continue / replan / end；达到最大轮次强制结束。"""
    query = state["user_query"]
    pending = list(state.get("pending_steps", []))
    completed = list(state.get("completed_steps", []))
    iteration = state.get("iteration_count", 0) + 1
    events = list(state.get("events", []))

    # 迭代上限：强制结束，保证最终一定产出 report + done
    if iteration >= MAX_ITERATIONS:
        logger.info({"event": "replanner_force_end", "iteration": iteration})
        events.append(make_event(
            "replan", agent=AGENT_NAME,
            content=f"已达到最大排查轮次（{MAX_ITERATIONS}），进入报告生成。",
        ))
        return {"should_end": True, "iteration_count": iteration, "events": events}

    p = parser if parser is not None else LlmParser()
    user_prompt = (
        f"用户故障描述：\n{query}\n\n"
        f"已完成步骤与结果：\n{_format_completed(completed) or '（暂无）'}\n\n"
        f"剩余待执行步骤：\n{chr(10).join(s.description for s in pending) or '（无）'}\n"
        f"已完成 {iteration} 轮排查。\n"
        f"请判断：continue（继续下一步）/ replan（给出新的待执行步骤）/ end（信息足够，生成报告）。"
    )
    decision = p.parse_replan(load_diagnostic_replan_prompt(), user_prompt)

    if decision is None:
        # 决策不可解析：有剩余步骤则按原计划继续，无剩余则结束，绝不死循环
        if pending:
            logger.warning({"event": "replanner_parse_failed_continue"})
            events.append(make_event("replan", agent=AGENT_NAME, content="重规划输出无法解析，按原计划继续。"))
            return {"iteration_count": iteration, "should_end": False, "events": events}
        logger.warning({"event": "replanner_parse_failed_end"})
        events.append(make_event("replan", agent=AGENT_NAME, content="重规划输出无法解析且无剩余步骤，进入报告生成。"))
        return {"should_end": True, "iteration_count": iteration, "events": events}

    if decision.action == "end":
        logger.info({"event": "replanner_end", "reason": log_safe_text(decision.reason)})
        events.append(make_event(
            "replan", agent=AGENT_NAME,
            content=decision.reason or "排查信息已足够，进入报告生成。",
        ))
        return {"should_end": True, "iteration_count": iteration, "events": events}

    if decision.action == "replan":
        new_steps = _dedupe_steps(decision.steps or [], completed)[:MAX_STEPS]
        if not new_steps:
            # 新计划为空（或全部与已完成重复）→ 结束
            logger.info({"event": "replanner_empty_new_plan", "reason": log_safe_text(decision.reason)})
            events.append(make_event("replan", agent=AGENT_NAME, content="新计划为空，进入报告生成。"))
            return {"should_end": True, "iteration_count": iteration, "events": events}
        logger.info({
            "event": "replanner_new_plan",
            "new_plan": log_safe_value([s.description for s in new_steps]),
            "reason": log_safe_text(decision.reason),
        })
        events.append(make_event(
            "replan", agent=AGENT_NAME,
            content=f"调整计划：{decision.reason}\n新计划：{'；'.join(s.description for s in new_steps)}",
        ))
        # 新计划整体替换待执行队列，executor 从第一项开始执行
        return {
            "pending_steps": new_steps,
            "iteration_count": iteration,
            "should_end": False,
            "events": events,
        }

    # continue
    if not pending:
        events.append(make_event("replan", agent=AGENT_NAME, content="计划已全部执行完，进入报告生成。"))
        return {"should_end": True, "iteration_count": iteration, "events": events}
    logger.info({"event": "replanner_continue", "reason": log_safe_text(decision.reason)})
    events.append(make_event("replan", agent=AGENT_NAME, content=decision.reason or "继续下一步排查。"))
    return {"iteration_count": iteration, "should_end": False, "events": events}


# ----------------------------------------------------------- reporter

def _fallback_report(query: str, completed: list[CompletedStep]) -> str:
    """LLM 报告生成失败时的结构化兜底报告（区分已确认事实 / 工具不可用 / 推测性建议）。"""
    confirmed = [c for c in completed if c.result.success]
    unavailable = [c for c in completed if not c.result.success]

    lines = [
        "## 诊断报告",
        "### 故障描述",
        query or "（未提供）",
        "### 排查过程",
    ]
    if completed:
        for i, c in enumerate(completed, start=1):
            if c.result.success:
                lines.append(f"{i}. {c.step.description}（工具 {c.step.tool}）：{c.result.content}")
            else:
                lines.append(
                    f"{i}. {c.step.description}（工具 {c.step.tool}）："
                    f"（工具不可用/调用失败：{c.result.safe_error_message}）"
                )
    else:
        lines.append("（无具体执行结果）")
    lines.append("### 故障原因")
    if confirmed:
        lines.append("基于以上已确认的排查数据，需要人工进一步分析定位。")
    else:
        lines.append("本次排查未获得有效数据，无法定位故障原因。")
    lines.append("### 处置建议")
    if unavailable:
        lines.append(f"- 以下步骤的工具不可用，建议稍后重试：{'、'.join(c.step.description for c in unavailable)}")
    lines.append("- 以上结论为推测性建议，建议联系人工客服进一步确认。")
    lines.append("### 参考资料")
    lines.append("（报告生成服务暂不可用，本报告由系统自动汇总）")
    return "\n".join(lines)


def reporter_node(state: dict, model: ChatModelLike | None = None) -> dict:
    """汇总已完成步骤生成 Markdown 诊断报告；LLM 失败时使用结构化兜底报告。

    模型获取（get_chat_model）与 invoke 同在保护范围内：任一环节失败
    都降级为 _fallback_report，保证最终事件流一定包含一个 report。
    """
    query = state["user_query"]
    completed = list(state.get("completed_steps", []))
    user_prompt = (
        f"用户故障描述：\n{query}\n\n"
        f"排查过程与结果：\n{_format_completed(completed) or '（无具体执行结果）'}\n\n"
        f"请按模板生成 Markdown 诊断报告。"
    )

    report = ""
    try:
        m = model if model is not None else get_chat_model()
        resp = m.invoke([
            SystemMessage(content=load_diagnostic_report_prompt()),
            HumanMessage(content=user_prompt),
        ])
        report = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        logger.error({
            "event": "reporter_error",
            **safe_exception_fields(e),
        })
        report = ""

    if not report.strip():
        report = _fallback_report(query, completed)

    events = list(state.get("events", []))
    events.append(make_event("report", agent=AGENT_NAME, content=report))

    logger.info({"event": "reporter_done", "report_length": len(report)})
    return {"final_report": report, "events": events}


__all__ = [
    "AGENT_NAME",
    "DiagnosticState",
    "initial_state",
    "planner_node",
    "executor_node",
    "replanner_node",
    "reporter_node",
]
