"""MCP Server 冒烟测试：验证 device_server 与 log_server 的协议握手、工具列表、工具调用。

用法：python scripts/mcp_smoke.py
不依赖 LLM，仅验证 MCP 协议层是否打通。
"""
import asyncio
import logging
import os
import sys

logging.getLogger("mcp").setLevel(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

DEVICE_SERVER = os.path.join(PROJECT_ROOT, "mcp_server", "device_server.py")
LOG_SERVER = os.path.join(PROJECT_ROOT, "mcp_server", "log_server.py")


async def probe(server_path: str, tool_name: str, arguments: dict) -> dict:
    params = StdioServerParameters(command=sys.executable, args=[server_path])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        tool_names = [t.name for t in tools.tools]
        result = await session.call_tool(tool_name, arguments)
        text = result.content[0].text if result.content else ""
        return {"tools": tool_names, "text": text}


async def main():
    errors = []

    dev = await probe(DEVICE_SERVER, "query_device_status", {"user_id": "123"})
    print("DEVICE_TOOLS:", dev["tools"])
    print("DEVICE_CALL_OK:", bool(dev["text"]))
    if not dev["text"]:
        errors.append("device_server: query_device_status returned empty")

    log = await probe(LOG_SERVER, "query_device_logs", {"user_id": "123", "days": 3})
    print("LOG_TOOLS:", log["tools"])
    print("LOG_CALL_OK:", bool(log["text"]))
    if not log["text"]:
        errors.append("log_server: query_device_logs returned empty")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nALL MCP SMOKE TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
