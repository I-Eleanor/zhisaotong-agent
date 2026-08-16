"""设备状态查询 MCP Server（stdio 传输）。

通过标准 MCP 协议把「设备运行状态查询」能力暴露给 Agent 或任意 MCP Host。
数据来源复用项目内的 CsvDeviceStatusService（从 CSV 读取设备数据）。

运行：python mcp_server/device_server.py
测试：python -m mcp_server.device_server   （或经 MCP 客户端 stdio 连接）
"""
import os
import sys

# 保证工程根在 sys.path，使 agent / utils 可导入（无论从哪个 cwd 启动）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from agent.services import create_device_status_service, create_user_id_service  # noqa: E402

mcp = FastMCP("device-status-server")


@mcp.tool()
def query_device_status(user_id: str) -> str:
    """查询指定用户设备的运行状态（覆盖率、清洁效率、耗材状态）。

    入参 user_id 为数字字符串（如 '123'）；为空时使用默认 mock 用户。
    """
    uid = user_id or create_user_id_service("mock").get_user_id()
    svc = create_device_status_service("csv")
    return svc.get_status(uid)


@mcp.tool()
def query_current_user() -> str:
    """获取当前默认用户 ID（mock 环境）。"""
    return create_user_id_service("mock").get_user_id()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
