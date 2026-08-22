"""兼容入口：诊断 Agent 已拆分至 agent/diagnostic/ 包。

包结构（schemas / parser / tool_router / nodes / graph / service）见 agent/diagnostic/__init__.py。
本模块保留旧导入路径，旧测试与调用方（orchestrator、scripts）无需修改。
"""
from agent.diagnostic import (
    AGENT_NAME,
    MAX_ITERATIONS,
    DiagnosticAgent,
    DiagnosticState,
    executor_node,
    get_diagnostic_agent,
    planner_node,
    replanner_node,
    reporter_node,
    reset_diagnostic_agent,
)

__all__ = [
    "AGENT_NAME",
    "MAX_ITERATIONS",
    "DiagnosticAgent",
    "DiagnosticState",
    "executor_node",
    "get_diagnostic_agent",
    "planner_node",
    "replanner_node",
    "reporter_node",
    "reset_diagnostic_agent",
]
