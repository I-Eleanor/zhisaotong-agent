"""诊断 Agent（Plan-Execute-Replan）。

架构模式：Plan-Execute-Replan（计划-执行-重规划），对标运维类诊断 Agent。
四个节点（LangGraph StateGraph 编排）：
    planner    根据故障描述生成结构化排查计划（JSON 步骤列表）
    executor   执行当前步骤，按步骤语义选择诊断工具获取数据
    replanner  根据执行结果决定 continue / replan / end
    reporter   汇总所有排查结果生成 Markdown 诊断报告

迭代上限 5 轮；每轮产出的事件累积在 state.events 中，run() 以流式方式逐个吐出。

事件类型（见 agent/events.py）：plan / step / replan / report / error / done
"""
import json
import re
from typing import Iterator, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from agent.events import AgentEvent, make_event
from agent.tools.diagnostic_tools import (
    query_device_status,
    query_error_code,
    query_maintenance,
    retrieve_knowledge,
    current_user_id,
)
from model.factory import get_chat_model
from utils.logger_handler import logger
from utils.prompt_loader import (
    load_diagnostic_plan_prompt,
    load_diagnostic_replan_prompt,
    load_diagnostic_report_prompt,
)

AGENT_NAME = "diagnostic"
MAX_ITERATIONS = 5


class DiagnosticState(TypedDict):
    """诊断流程状态（LangGraph 状态模式：TypedDict）。

    各字段均为「替换」语义的 channel；节点每次返回完整的累积列表（如 events），
    因此无需额外 reducer 即可在节点间传递上下文。
    """

    user_query: str
    plan: list
    current_step_index: int
    execution_results: list
    iteration_count: int
    final_report: str
    should_end: bool
    events: list


# ----------------------------------------------------------- JSON 解析工具

def _extract_json(text: str):
    """从 LLM 输出中尽力解析出 JSON（兼容 ```json 代码块与裸 JSON）。"""
    if not text:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 退而求其次：截取第一个 [ 或 { 到最后一个 ] 或 }
    arr = re.search(r"\[.*\]", text, re.DOTALL)
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    for match in (arr, obj):
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None


def _llm_json(system_prompt: str, user_prompt: str):
    """调用 LLM 并解析为 JSON；失败返回 None。"""
    try:
        model = get_chat_model()
        resp = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return _extract_json(content)
    except Exception as e:
        logger.error({
            "event": "diagnostic_llm_error",
            "error_type": type(e).__name__,
            "error_msg": str(e),
        })
        return None


# ----------------------------------------------------------- 节点实现

def planner_node(state: DiagnosticState) -> dict:
    query = state["user_query"]
    plan_prompt = load_diagnostic_plan_prompt()
    user_prompt = f"用户故障描述：\n{query}\n\n请生成本次排查计划（JSON 数组，最多 5 个步骤）。"

    parsed = _llm_json(plan_prompt, user_prompt)
    if isinstance(parsed, dict) and isinstance(parsed.get("plan"), list):
        steps = [str(s) for s in parsed["plan"]]
    elif isinstance(parsed, list):
        steps = [str(s) for s in parsed]
    else:
        # 兜底：给一份固定计划，保证流程不中断
        logger.warning({"event": "planner_fallback", "query": query})
        steps = ["查询设备运行状态", "检索故障排除手册", "检索维护保养建议"]

    steps = steps[:MAX_ITERATIONS]
    events = list(state.get("events", []))
    events.append(make_event("plan", agent=AGENT_NAME, content="", steps=steps))

    logger.info({"event": "planner_done", "steps": steps})
    return {
        "plan": steps,
        "current_step_index": 0,
        "iteration_count": state.get("iteration_count", 0),
        "should_end": False,
        "events": events,
    }


def _select_tool(step: str, query: str):
    """根据步骤语义选择诊断工具与入参。"""
    s = step.lower()
    # 错误码
    if "错误码" in step or "error code" in s or re.search(r"\be\d{1,3}\b", step, re.IGNORECASE):
        code = re.search(r"\be\d{1,3}\b", query, re.IGNORECASE)
        arg = code.group(0).upper() if code else step
        return query_error_code, arg, "错误码查询"
    # 维护 / 保养
    if "维护" in step or "保养" in step or "maintenance" in s:
        return query_maintenance, step, "维护建议查询"
    # 设备状态 / 电量 / 耗材 / 清洁效率
    if any(k in step for k in ("设备状态", "电量", "耗材", "清洁效率", "覆盖率", "运行状态")):
        user_id = current_user_id()
        return query_device_status, user_id, f"设备状态查询(user_id={user_id})"
    # 默认全库检索
    return retrieve_knowledge, query, "全库知识检索"


def executor_node(state: DiagnosticState) -> dict:
    plan = state["plan"]
    idx = state["current_step_index"]
    query = state["user_query"]

    if idx >= len(plan):
        # 计划已执行完，直接进入报告
        events = list(state.get("events", []))
        return {"should_end": True, "events": events}

    step = plan[idx]
    tool, arg, label = _select_tool(step, query)
    logger.info({"event": "executor_step", "index": idx, "step": step, "tool": tool.name, "arg": arg})

    try:
        result = tool.invoke(arg)
    except Exception as e:
        result = f"工具{getattr(tool, 'name', 'unknown')}执行失败：{str(e)}"

    events = list(state.get("events", []))
    events.append(make_event(
        "step", agent=AGENT_NAME, content=str(result),
        index=idx + 1, description=step, tool=label,
    ))

    results = list(state.get("execution_results", []))
    results.append(f"【步骤{idx + 1}：{step}】\n{result}")

    return {"execution_results": results, "events": events}


def replanner_node(state: DiagnosticState) -> dict:
    plan = state["plan"]
    results = state["execution_results"]
    idx = state["current_step_index"]
    iteration = state.get("iteration_count", 0) + 1
    query = state["user_query"]

    events = list(state.get("events", []))

    # 迭代上限：强制结束
    if iteration >= MAX_ITERATIONS:
        logger.info({"event": "replanner_force_end", "iteration": iteration})
        events.append(make_event("replan", agent=AGENT_NAME,
                                 content=f"已达到最大排查轮次（{MAX_ITERATIONS}），进入报告生成。"))
        return {"should_end": True, "iteration_count": iteration, "events": events}

    replan_prompt = load_diagnostic_replan_prompt()
    user_prompt = (
        f"用户故障描述：\n{query}\n\n"
        f"当前排查计划：\n{json.dumps(plan, ensure_ascii=False)}\n\n"
        f"已执行步骤与结果：\n{chr(10).join(results)}\n\n"
        f"当前执行到第 {idx + 1} 步，已完成 {iteration} 轮排查。\n"
        f"请判断：continue（继续下一步）/ replan（修改计划，给出新 JSON 计划）/ end（信息足够，生成报告）。"
    )

    parsed = _llm_json(replan_prompt, user_prompt)
    action = "continue"
    new_plan = None
    reason = ""
    if isinstance(parsed, dict):
        action = str(parsed.get("action", "continue")).lower()
        new_plan = parsed.get("plan") or parsed.get("new_plan")
        reason = str(parsed.get("reason", ""))

    if action == "end":
        logger.info({"event": "replanner_end", "reason": reason})
        events.append(make_event("replan", agent=AGENT_NAME, content=reason or "排查信息已足够，进入报告生成。"))
        return {"should_end": True, "iteration_count": iteration, "events": events}

    if action == "replan" and isinstance(new_plan, list) and new_plan:
        new_plan = [str(s) for s in new_plan][:MAX_ITERATIONS]
        logger.info({"event": "replanner_new_plan", "new_plan": new_plan, "reason": reason})
        events.append(make_event("replan", agent=AGENT_NAME,
                                 content=f"调整计划：{reason}\n新计划：{'；'.join(new_plan)}"))
        # 从当前步之后推进，新插入的步骤（通常在末尾）会随后执行
        return {
            "plan": new_plan,
            "current_step_index": min(idx + 1, len(new_plan) - 1),
            "iteration_count": iteration,
            "should_end": False,
            "events": events,
        }

    # 默认 continue
    logger.info({"event": "replanner_continue", "reason": reason})
    events.append(make_event("replan", agent=AGENT_NAME, content=reason or "继续下一步排查。"))
    return {
        "current_step_index": idx + 1,
        "iteration_count": iteration,
        "should_end": False,
        "events": events,
    }


def reporter_node(state: DiagnosticState) -> dict:
    query = state["user_query"]
    results = state["execution_results"]
    report_prompt = load_diagnostic_report_prompt()
    user_prompt = (
        f"用户故障描述：\n{query}\n\n"
        f"排查过程与结果：\n{chr(10).join(results) if results else '（无具体执行结果）'}\n\n"
        f"请按模板生成 Markdown 诊断报告。"
    )

    model = get_chat_model()
    try:
        resp = model.invoke([
            SystemMessage(content=report_prompt),
            HumanMessage(content=user_prompt),
        ])
        report = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        logger.error({"event": "reporter_error", "error_msg": str(e)})
        report = f"诊断报告生成失败：{str(e)}"

    if not report.strip():
        report = "未能生成诊断报告，请稍后重试。"

    events = list(state.get("events", []))
    events.append(make_event("report", agent=AGENT_NAME, content=report))

    logger.info({"event": "reporter_done", "report_length": len(report)})
    return {"final_report": report, "events": events}


# ----------------------------------------------------------- 图构建

def _build_graph():
    workflow = StateGraph(DiagnosticState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replanner")
    workflow.add_conditional_edges(
        "replanner",
        lambda s: "reporter" if s.get("should_end") else "executor",
        {"reporter": "reporter", "executor": "executor"},
    )
    workflow.add_edge("reporter", END)
    return workflow.compile()


class DiagnosticAgent:
    def __init__(self):
        self.graph = _build_graph()

    def run(self, user_query: str) -> Iterator[AgentEvent]:
        """流式执行诊断流程，逐个产出 plan/step/replan/report 事件。"""
        initial = {
            "user_query": user_query,
            "plan": [],
            "current_step_index": 0,
            "execution_results": [],
            "iteration_count": 0,
            "final_report": "",
            "should_end": False,
            "events": [],
        }
        seen = 0
        try:
            for snapshot in self.graph.stream(initial, stream_mode="values"):
                evs = snapshot.get("events", []) or []
                for ev in evs[seen:]:
                    yield ev
                seen = len(evs)
        except Exception as e:
            logger.error({
                "event": "diagnostic_agent_error",
                "query": user_query,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            })
            yield make_event("error", agent=AGENT_NAME, content=f"诊断流程执行失败：{str(e)}")
        yield make_event("done", agent=AGENT_NAME)


_diagnostic_agent = None


def get_diagnostic_agent() -> DiagnosticAgent:
    global _diagnostic_agent
    if _diagnostic_agent is None:
        _diagnostic_agent = DiagnosticAgent()
    return _diagnostic_agent


def reset_diagnostic_agent() -> None:
    global _diagnostic_agent
    _diagnostic_agent = None


if __name__ == '__main__':
    agent = DiagnosticAgent()
    for ev in agent.run("我的扫地机器人最近清洁效率很低"):
        print(ev)
