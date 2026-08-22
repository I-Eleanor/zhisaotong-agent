"""诊断 Agent 包（Plan-Execute-Replan）。

模块职责：
    schemas.py     Pydantic 数据结构与工具白名单
    parser.py      LLM structured output 兼容解析层
    tool_router.py 工具白名单、参数校验与调用
    nodes.py       planner / executor / replanner / reporter 节点逻辑
    graph.py       LangGraph 图连接
    service.py     对外 DiagnosticAgent 服务

旧导入路径 agent.diagnostic_agent 仍可用（兼容入口）。
"""
from agent.diagnostic.graph import build_diagnostic_graph
from agent.diagnostic.nodes import (
    AGENT_NAME,
    DiagnosticState,
    executor_node,
    initial_state,
    planner_node,
    replanner_node,
    reporter_node,
)
from agent.diagnostic.parser import LlmParser, extract_json
from agent.diagnostic.schemas import (
    ALLOWED_TOOLS,
    MAX_ITERATIONS,
    MAX_STEPS,
    CompletedStep,
    DiagnosticPlan,
    DiagnosticStep,
    ReplanDecision,
    StepResult,
)
from agent.diagnostic.service import (
    DiagnosticAgent,
    get_diagnostic_agent,
    reset_diagnostic_agent,
)
from agent.diagnostic.tool_router import ToolRouter, ToolSpec, build_default_tool_specs

__all__ = [
    "AGENT_NAME",
    "ALLOWED_TOOLS",
    "MAX_ITERATIONS",
    "MAX_STEPS",
    "CompletedStep",
    "DiagnosticAgent",
    "DiagnosticPlan",
    "DiagnosticState",
    "DiagnosticStep",
    "LlmParser",
    "ReplanDecision",
    "StepResult",
    "ToolRouter",
    "ToolSpec",
    "build_default_tool_specs",
    "build_diagnostic_graph",
    "executor_node",
    "extract_json",
    "get_diagnostic_agent",
    "initial_state",
    "planner_node",
    "replanner_node",
    "reporter_node",
    "reset_diagnostic_agent",
]
