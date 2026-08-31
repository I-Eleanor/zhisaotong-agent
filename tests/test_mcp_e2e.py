"""主链路端到端测试（真实 MCP 子进程）。

证明实际执行链路为：
    DiagnosticAgent → ToolRouter → MCP Client → stdio MCP Server → CSV / Mock

覆盖：
- Agent 执行了设备状态工具与日志工具（结果进入 completed_steps）
- 最终报告包含设备数据
- MCP 不可用（Server 路径错误）→ 步骤安全降级（TOOL_UNAVAILABLE）、
  报告含"实时设备数据暂时不可用"安全文案、事件流零内部信息泄漏
- 日志工具参数 days 由 Planner 产出并被 ToolRouter 正确传递（1-30 归一）
"""
import json

import pytest

from agent.diagnostic.schemas import (
    DiagnosticPlan,
    DiagnosticStep,
    ReplanDecision,
    StepResult,
)
from agent.diagnostic.tool_router import ToolRouter
from agent.services.device_data_provider import McpDeviceDataProvider
from agent.tools import diagnostic_tools as dt


class StubParser:
    """按脚本返回计划/决策的解析替身。"""

    def __init__(self, plan, decisions=None):
        self.plan = plan
        self.decisions = list(decisions or [])

    def parse_plan(self, system_prompt, user_prompt):
        return self.plan

    def parse_replan(self, system_prompt, user_prompt):
        # 第一步后 continue 执行第二步，随后默认 end 进入报告
        return self.decisions.pop(0) if self.decisions else ReplanDecision(action="end", reason="信息已足够")


class StubReportModel:
    """reporter 假模型：把排查过程原样回填为报告（便于断言报告含设备数据）。"""

    def invoke(self, messages, **kw):
        captured.append(messages)

        class R:
            content = messages[-1].content if hasattr(messages[-1], "content") else str(messages)
        return R()


captured: list = []


@pytest.fixture
def mcp_provider(monkeypatch):
    """把默认 Provider 切换为 MCP（覆盖 conftest 的 direct 注入），验证真实调用链。"""
    monkeypatch.setenv("DEVICE_DATA_PROVIDER", "mcp")
    dt.reset_device_data_provider()
    yield McpDeviceDataProvider()
    dt.reset_device_data_provider()


def _step(desc, tool="query_device_status", args=None):
    return DiagnosticStep(description=desc, tool=tool, arguments=args or {})


def test_tool_router_executes_device_status_over_mcp(mcp_provider):
    """ToolRouter → MCP Client → device_server：状态查询成功并进入 StepResult。"""
    router = ToolRouter()
    result = router.execute(_step("查询设备运行状态", args={"user_id": "1001"}), user_query="清洁效率低")
    assert result.success is True, "MCP 链路下的设备状态工具应成功"
    assert "设备ID：device-1001" in result.content
    assert "覆盖率" in result.content or "清洁效率" in result.content


def test_tool_router_executes_device_logs_over_mcp(mcp_provider):
    """ToolRouter → MCP Client → log_server：日志工具（含 days 参数）成功。"""
    router = ToolRouter()
    result = router.execute(
        _step("查询最近运行日志", tool="query_device_logs", args={"user_id": "1001", "days": "3"}),
        user_query="最近经常卡顿",
    )
    assert result.success is True
    assert "设备ID：device-1001" in result.content
    assert "运行日志" in result.content


def test_end_to_end_diagnostic_agent_runs_status_and_logs_over_mcp(mcp_provider):
    """完整链路：DiagnosticAgent 同时执行状态与日志工具，报告含设备数据。"""
    from agent.diagnostic.service import DiagnosticAgent

    plan = DiagnosticPlan(steps=[
        _step("查询设备运行状态", args={"user_id": "1001"}),
        _step("查询最近运行日志", tool="query_device_logs", args={"user_id": "1001", "days": "3"}),
    ])
    parser = StubParser(plan=plan, decisions=[ReplanDecision(action="continue", reason="继续下一步")])
    agent = DiagnosticAgent(parser=parser, tool_router=ToolRouter(), model=StubReportModel())

    events = list(agent.run("最近清洁效率很低"))
    step_events = [e for e in events if e["type"] == "step"]

    assert len(step_events) == 2, "状态与日志两个步骤都应执行"
    tools = [e["data"]["tool"] for e in step_events]
    assert "query_device_status" in tools and "query_device_logs" in tools
    assert all(e["data"].get("error_code", "") == "" for e in step_events), "两步都应成功"

    report = [e for e in events if e["type"] == "report"][0]["content"]
    assert "设备ID：device-1001" in report, "报告应包含经 MCP 查询到的设备数据"
    assert events[-1]["type"] == "done"


def test_end_to_end_mcp_failure_degrades_to_safe_step_result(mcp_provider, monkeypatch):
    """MCP 不可用：步骤失败 → TOOL_UNAVAILABLE + 固定安全文案，报告零内部信息泄漏。"""
    from agent.diagnostic.service import DiagnosticAgent

    # 指向不存在的 Server：模拟 MCP Server 启动失败
    broken = McpDeviceDataProvider(
        device_client=dt_module_client("<PROJECT_ROOT>/mcp_server/__not_exist__.py"),
        log_client=log_module_client("<PROJECT_ROOT>/mcp_server/__not_exist__.py"),
    )
    monkeypatch.setattr(dt, "_device_data_provider", broken, raising=False)

    plan = DiagnosticPlan(steps=[
        _step("查询设备运行状态", args={"user_id": "1001"}),
        _step("查询最近运行日志", tool="query_device_logs", args={"user_id": "1001", "days": "3"}),
    ])
    parser = StubParser(plan=plan, decisions=[ReplanDecision(action="continue", reason="继续下一步")])
    agent = DiagnosticAgent(parser=parser, tool_router=ToolRouter(), model=StubReportModel())

    events = list(agent.run("扫地机不工作"))
    step_events = [e for e in events if e["type"] == "step"]

    assert len(step_events) == 2
    for event in step_events:
        assert event["data"]["error_code"] == "TOOL_UNAVAILABLE", "MCP 失败应统一为工具不可用"
        assert "实时设备数据暂时不可用" in event["content"], "用户只看到固定安全文案"

    report = [e for e in events if e["type"] == "report"][0]["content"]
    assert "实时设备数据暂时不可用" in report or "工具不可用" in report

    serialized = json.dumps(events, ensure_ascii=False, default=str)
    for leaked in ("__not_exist__", "Traceback", "McpServerStartupError", "mcp_server"):
        assert leaked not in serialized, f"事件流不得泄漏内部信息：{leaked}"


def test_end_to_end_unknown_user_degrades_to_service_unavailable(mcp_provider):
    """CSV 中没有该用户数据：经 MCP 哨兵文本 → SERVICE_UNAVAILABLE（不当作成功事实）。"""
    router = ToolRouter()
    result = router.execute(_step("查状态", args={"user_id": "999999"}), user_query="不工作")
    assert result.success is False
    assert result.error_code == "SERVICE_UNAVAILABLE"
    assert "999999" not in result.safe_error_message, "安全文案不得回显用户标识"


def test_log_tool_days_argument_is_normalized_by_router(mcp_provider):
    """Planner 产出的 days 越界时由工具层归一化到 1-30，不影响调用成功。"""
    router = ToolRouter()
    result = router.execute(
        _step("查询最近运行日志", tool="query_device_logs", args={"user_id": "1001", "days": "999"}),
        user_query="最近异常",
    )
    assert result.success is True
    assert "运行日志" in result.content


# ----------------------------------------------------------------- 辅助：构造坏客户端
def dt_module_client(bad_path):
    from agent.mcp_client.device_client import DeviceMcpClient
    return DeviceMcpClient(server_path=bad_path)


def log_module_client(bad_path):
    from agent.mcp_client.log_client import LogMcpClient
    return LogMcpClient(server_path=bad_path)


def test_step_result_success_shape_for_mcp_content():
    """成功 StepResult 只携带内容，不携带错误码（保持既有协议）。"""
    result = StepResult(success=True, content="设备ID：device-1001")
    assert result.error_code == "" and result.safe_error_message == ""
