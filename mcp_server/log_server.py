"""设备运行日志查询 MCP Server（stdio 传输）。

通过标准 MCP 协议把「设备运行日志查询」能力暴露给 Agent 或任意 MCP Host。
数据来源为 MockDeviceLogService：基于用户 ID 派生可复现的模拟日志
（含错误码、告警、运行时长等，格式 {timestamp, level, event, device_id, message}）。

运行：python mcp_server/log_server.py
"""
import os
import sys

# 保证工程根在 sys.path，使 agent / utils 可导入（无论从哪个 cwd 启动）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP
from agent.services import create_device_log_service, create_user_id_service

mcp = FastMCP("device-log-server")


@mcp.tool()
def query_device_logs(user_id: str, days: int = 7) -> str:
    """查询指定用户设备最近 days 天的运行日志（含错误码、告警、运行时长等）。

    入参 user_id 为数字字符串；days 为整数，默认 7。
    """
    uid = user_id or create_user_id_service("mock").get_user_id()
    svc = create_device_log_service("mock")
    return svc.get_logs(uid, days)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
