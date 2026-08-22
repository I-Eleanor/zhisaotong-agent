"""诊断流程图（LangGraph StateGraph 连接）。

planner → executor → replanner →（should_end ? reporter : executor）→ END
依赖（parser / tool_router / model）通过 partial 注入节点，便于测试替换。
"""
from functools import partial

from langgraph.graph import END, StateGraph

from agent.diagnostic.nodes import (
    DiagnosticState,
    executor_node,
    planner_node,
    replanner_node,
    reporter_node,
)
from agent.diagnostic.parser import LlmParser
from agent.diagnostic.tool_router import ToolRouter


def build_diagnostic_graph(
    parser: LlmParser | None = None,
    tool_router: ToolRouter | None = None,
    model=None,
):
    """编译诊断流程图；三个依赖均可注入，None 时节点内懒创建默认实现。"""
    workflow = StateGraph(DiagnosticState)
    workflow.add_node("planner", partial(planner_node, parser=parser))
    workflow.add_node("executor", partial(executor_node, tool_router=tool_router))
    workflow.add_node("replanner", partial(replanner_node, parser=parser))
    workflow.add_node("reporter", partial(reporter_node, model=model))

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
