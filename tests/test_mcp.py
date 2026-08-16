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


def test_log_mcp_server():
    names, text = asyncio.run(_probe(LOG_SERVER, "query_device_logs", {"user_id": "123", "days": 3}))
    assert "query_device_logs" in names
    assert text, "日志查询应返回非空结果"
