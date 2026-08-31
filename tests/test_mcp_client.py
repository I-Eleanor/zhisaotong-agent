"""MCP Client 适配层测试（真实 stdio 子进程 + 失败场景）。

覆盖：
- device_server / log_server 握手、工具发现、成功调用、返回非空
- 哨兵"数据不可用"文本 → ServiceUnavailableError（CSV 无该用户数据时）
- 失败场景：Server 路径不存在、Server 立即退出、工具不存在、调用超时、
  返回空结果 —— 全部统一为 MCPConnectionError 族，且不泄漏内部信息
"""
import sys
import textwrap

import pytest

from agent.mcp_client.client import DEVICE_SERVER_PATH, LOG_SERVER_PATH, McpToolClient
from agent.mcp_client.device_client import DeviceMcpClient
from agent.mcp_client.exceptions import (
    McpServerStartupError,
    McpTimeoutError,
    McpToolCallError,
)
from agent.mcp_client.log_client import LogMcpClient, normalize_days
from utils.exceptions import MCPConnectionError, ServiceUnavailableError


# ----------------------------------------------------------------- 成功路径
def test_device_client_query_status_returns_device_status():
    text = DeviceMcpClient().query_status("1001")
    assert "设备ID：device-1001" in text, "返回应体现用户绑定设备"
    assert "覆盖率" in text or "清洁效率" in text, "应包含 CSV 模拟设备状态"


def test_repeated_calls_each_spawn_independent_server():
    """连续调用证明每次调用都是独立的「启动→握手→调用→关闭」生命周期，无泄漏。"""
    client = McpToolClient(DEVICE_SERVER_PATH)
    first = client.call("query_device_status", {"user_id": "1001"})
    second = client.call("query_device_status", {"user_id": "1002"})
    assert first and second
    assert "device-1001" in first and "device-1002" in second


def test_call_inside_running_loop_degrades_safely():
    """在运行中事件循环里同步调用：不抛 RuntimeError，而是安全降级为 MCP 连接失败。"""
    import asyncio

    client = McpToolClient(DEVICE_SERVER_PATH)

    def inside_loop():
        return client.call("query_device_status", {"user_id": "1001"})

    with pytest.raises(McpServerStartupError):
        asyncio.run(_async_call(inside_loop))


async def _async_call(fn):
    return fn()


def test_log_client_query_logs_returns_logs():
    text = LogMcpClient().query_logs("1001", days=3)
    assert "设备ID：device-1001" in text
    assert "运行日志" in text
    assert text.strip()


def test_mcp_client_can_discover_and_call_device_tool():
    client = McpToolClient(DEVICE_SERVER_PATH)
    text = client.call("query_device_status", {"user_id": "1001"})
    assert text, "工具调用结果非空"
    assert "设备ID" in text


def test_mcp_client_can_discover_and_call_log_tool():
    client = McpToolClient(LOG_SERVER_PATH)
    text = client.call("query_device_logs", {"user_id": "1001", "days": 2})
    assert text, "日志工具调用结果非空"


def test_unavailable_sentinel_maps_to_service_unavailable():
    """CSV 中没有该用户数据 → Server 返回哨兵文本 → Client 还原为服务不可用。"""
    client = McpToolClient(DEVICE_SERVER_PATH)
    with pytest.raises(ServiceUnavailableError):
        client.call("query_device_status", {"user_id": "999999"})


def test_unknown_user_short_circuits_without_subprocess():
    """未知用户：Client 在注册表校验处失败，不启动子进程。"""
    broken_client = McpToolClient("<不存在的服务路径>")
    with pytest.raises(ServiceUnavailableError):
        DeviceMcpClient(server_path=broken_client.server_path).query_status("999999")


# ----------------------------------------------------------------- days 归一化
@pytest.mark.parametrize("value,expected", [(1, 1), (7, 7), (30, 30), (0, 1), (999, 30), ("5", 5), ("bad", 7)])
def test_normalize_days_clamps_to_1_30(value, expected):
    assert normalize_days(value) == expected


# ----------------------------------------------------------------- 失败场景
def test_server_path_missing_raises_startup_error():
    client = McpToolClient("<PROJECT_ROOT>/mcp_server/__not_exist__.py")
    with pytest.raises(McpServerStartupError):
        client.call("query_device_status", {"user_id": "1001"})


def test_server_exits_immediately_raises_startup_error(tmp_path):
    server = tmp_path / "exit_server.py"
    server.write_text(textwrap.dedent("""
        import sys
        sys.exit(3)
    """), encoding="utf-8")
    client = McpToolClient(str(server))
    with pytest.raises(McpServerStartupError):
        client.call("any_tool", {})


def test_unknown_tool_name_raises_tool_call_error():
    client = McpToolClient(DEVICE_SERVER_PATH)
    with pytest.raises(McpToolCallError):
        client.call("tool_that_does_not_exist", {"user_id": "1001"})


def test_call_timeout_raises_timeout_error():
    """超时：握手阶段即到期，统一为 McpTimeoutError。"""
    client = McpToolClient(DEVICE_SERVER_PATH, timeout_seconds=0.001)
    with pytest.raises(McpTimeoutError):
        client.call("query_device_status", {"user_id": "1001"})


def test_empty_result_raises_tool_call_error(tmp_path):
    """Server 返回空内容：视为工具调用失败（不把空结果当成功事实）。"""
    server = tmp_path / "empty_server.py"
    server.write_text(textwrap.dedent("""
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("empty-server")

        @mcp.tool()
        def empty_tool() -> str:
            return ""

        mcp.run(transport="stdio")
    """), encoding="utf-8")
    client = McpToolClient(str(server))
    with pytest.raises(McpToolCallError):
        client.call("empty_tool", {})


def test_slow_server_tool_raises_timeout_error(tmp_path):
    """Server 工具执行超时（连接已建立、工具阻塞）：同样归入超时失败。"""
    server = tmp_path / "slow_server.py"
    server.write_text(textwrap.dedent("""
        import time
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("slow-server")

        @mcp.tool()
        def slow_tool() -> str:
            time.sleep(30)
            return "done"

        mcp.run(transport="stdio")
    """), encoding="utf-8")
    client = McpToolClient(str(server), timeout_seconds=1.5)
    with pytest.raises(McpTimeoutError):
        client.call("slow_tool", {})


# ----------------------------------------------------------------- 安全语义
def test_connection_errors_share_mcp_stage_and_error_code():
    """所有 MCP 失败：stage=mcp、error_code=TOOL_UNAVAILABLE（统一降级口径）。"""
    client = McpToolClient("<PROJECT_ROOT>/mcp_server/__not_exist__.py")
    with pytest.raises(MCPConnectionError) as exc_info:
        client.call("query_device_status", {"user_id": "1001"})
    exc = exc_info.value
    assert exc.stage == "mcp"
    assert exc.error_code == "TOOL_UNAVAILABLE"


def test_failure_message_contains_no_absolute_paths_or_python_exe(tmp_path):
    """失败异常消息不得携带子进程命令 / 本地绝对路径（只进日志，但不应泄漏细节）。"""
    server = tmp_path / "exit_server.py"
    server.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    client = McpToolClient(str(server))
    with pytest.raises(McpServerStartupError) as exc_info:
        client.call("any_tool", {})
    message = str(exc_info.value)
    assert str(tmp_path) not in message
    assert sys.executable not in message
