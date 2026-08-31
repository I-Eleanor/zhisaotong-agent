"""MCP Server 冒烟测试：验证 device_server 与 log_server 的协议握手、工具列表、工具调用。

用法：python scripts/mcp_smoke.py
不依赖 LLM，仅验证 MCP 协议层是否打通。

使用已注册演示用户（1001）——未注册用户会被服务端归属校验拦截并返回
固定的"数据不可用"哨兵文本，无法验证真实数据链路；脚本同时校验返回值
不是哨兵文本，避免"拿到不可用文案也算通过"的假阳性。
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

# 已注册演示用户（见 agent/services/device_registry.py）
DEMO_USER_ID = "1001"
UNAVAILABLE_MARKERS = ("数据暂时不可用",)


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

    dev = await probe(DEVICE_SERVER, "query_device_status", {"user_id": DEMO_USER_ID})
    print("DEVICE_TOOLS:", dev["tools"])
    print("DEVICE_CALL_OK:", bool(dev["text"]))
    if "query_device_status" not in dev["tools"]:
        errors.append("device_server: query_device_status 未在工具列表中")
    if not dev["text"]:
        errors.append("device_server: query_device_status returned empty")
    if any(marker in dev["text"] for marker in UNAVAILABLE_MARKERS):
        errors.append("device_server: 返回了数据不可用哨兵文本（未真正取到设备状态）")
    if "设备ID：device-1001" not in dev["text"]:
        errors.append("device_server: 返回缺少用户绑定设备标识")

    log = await probe(LOG_SERVER, "query_device_logs", {"user_id": DEMO_USER_ID, "days": 3})
    print("LOG_TOOLS:", log["tools"])
    print("LOG_CALL_OK:", bool(log["text"]))
    if "query_device_logs" not in log["tools"]:
        errors.append("log_server: query_device_logs 未在工具列表中")
    if not log["text"]:
        errors.append("log_server: query_device_logs returned empty")
    if any(marker in log["text"] for marker in UNAVAILABLE_MARKERS):
        errors.append("log_server: 返回了数据不可用哨兵文本（未真正取到运行日志）")
    if "运行日志" not in log["text"]:
        errors.append("log_server: 返回缺少日志内容")

    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("\nALL MCP SMOKE TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
