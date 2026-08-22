"""MCP Server 测试：经 stdio 启动两个 MCP server，验证握手、工具列表与工具调用。

device_server / log_server 复用 CSV/mock 服务，不依赖 LLM，可离线运行。
"""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE_SERVER = os.path.join(PROJECT_ROOT, "mcp_server", "device_server.py")
LOG_SERVER = os.path.join(PROJECT_ROOT, "mcp_server", "log_server.py")


async def _probe(server_path: str, tool_name: str, arguments: dict):
    params = StdioServerParameters(command=sys.executable, args=[server_path])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        tool_names = [t.name for t in tools.tools]
        result = await session.call_tool(tool_name, arguments)
        text = result.content[0].text if result.content else ""
        return tool_names, text


def test_device_mcp_server():
    names, text = asyncio.run(_probe(DEVICE_SERVER, "query_device_status", {"user_id": "123"}))
    assert "query_device_status" in names
    assert "query_current_user" in names
    assert isinstance(text, str)


def test_device_mcp_unavailable_no_internal_info_leak():
    """设备数据不可用（用户不存在）时：MCP 返回固定安全文本，不泄漏异常原文 / 用户 ID。"""
    names, text = asyncio.run(_probe(DEVICE_SERVER, "query_device_status", {"user_id": "999999"}))
    assert "query_device_status" in names
    assert text == "设备状态数据暂时不可用，请稍后重试。", "应返回固定安全文本"
    assert "999999" not in text, "不得回显用户 ID"
    assert "未查询到" not in text and "Error" not in text, "不得泄漏内部异常原文"


def test_log_mcp_server():
    names, text = asyncio.run(_probe(LOG_SERVER, "query_device_logs", {"user_id": "123", "days": 3}))
    assert "query_device_logs" in names
    assert text, "日志查询应返回非空结果"


# ----------------------------------------------------------------- MCP 降级响应白名单（P1-11）
def test_mcp_fallback_response_whitelist_controls_tool_name():
    """MCP 降级响应：白名单工具名保留；任意文本（含密钥/路径）不得拼入客户端文案。"""
    from utils.resilience import mcp_fallback_response

    secret = "sk-MCPFALL-998877665544"
    abs_path = "D:\\secret\\mcp\\device.bin"
    leaky = f"{secret} {abs_path}"

    # 白名单内：受控工具名保留，异常细节不进文案
    ok = mcp_fallback_response("query_device_status", RuntimeError(f"内部错误 {secret} {abs_path}"))
    assert "query_device_status" in ok
    assert "实时设备数据不可用" in ok
    assert secret not in ok and abs_path not in ok, "异常细节不得进入客户端文案"

    # 白名单外（用户可控字符串）：退化为不含工具名的通用文案
    bad = mcp_fallback_response(leaky, RuntimeError("内部错误"))
    assert "实时设备数据不可用" in bad
    assert leaky not in bad, "任意 tool_name 不得拼入客户端文案"
    assert secret not in bad and abs_path not in bad


# ----------------------------------------------------------------- MCP 启动/构造失败安全（P1-12）
def test_mcp_tool_construction_failure_returns_safe_text(monkeypatch, caplog):
    """MCP 工具内服务构造失败：客户端只收固定安全文本，异常细节（密钥/路径）不泄漏。"""
    import logging

    from mcp_server import device_server

    secret = "sk-MCPINIT-998877665544"
    abs_path = "D:\\secret\\mcp\\construct.bin"

    def boom(provider: str):
        raise RuntimeError(f"服务构造失败 api_key={secret} {abs_path}")

    monkeypatch.setattr(device_server, "create_device_status_service", boom)

    with caplog.at_level(logging.ERROR, logger="agent"):
        result = device_server.query_device_status("123")

    assert result == "设备状态数据暂时不可用，请稍后重试。", "应返回固定安全文本"
    for leaked in (secret, abs_path, "RuntimeError", "服务构造失败"):
        assert leaked not in result, f"MCP 响应不得泄漏：{leaked[:30]}"

    entries = [r.msg for r in caplog.records
               if isinstance(r.msg, dict) and r.msg.get("event") == "mcp_device_status_error"]
    assert entries, "应记录构造失败事件"
    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter
    json_outs = [JsonFormatter().format(r) for r in caplog.records]
    console_outs = [CONSOLE_FORMAT.format(r) for r in caplog.records]
    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert secret not in combined, f"{label}：MCP 日志不得泄漏密钥"
        assert abs_path not in combined, f"{label}：MCP 日志不得泄漏路径"


def test_mcp_log_server_failure_returns_safe_text(monkeypatch, caplog):
    """log_server 工具失败：客户端只收固定安全文本，异常细节不泄漏。"""
    import logging

    from mcp_server import log_server

    secret = "sk-MCPLOG-998877665544"

    def boom(provider: str):
        raise RuntimeError(f"日志服务失败 api_key={secret}")

    monkeypatch.setattr(log_server, "create_device_log_service", boom)

    with caplog.at_level(logging.ERROR, logger="agent"):
        result = log_server.query_device_logs("123", days=3)

    assert result == "设备日志数据暂时不可用，请稍后重试。", "应返回固定安全文本"
    assert secret not in result and "RuntimeError" not in result
