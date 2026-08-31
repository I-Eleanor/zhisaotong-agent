"""设备状态 MCP Client：user_id → 默认 device_id → device_server.query_device_status。

调用前先经设备注册表校验用户-设备归属（客户端侧快速失败，未知用户
不发起子进程），再经 MCP 协议调用 device_server（服务端二次校验归属）。
工具参数保持 user_id 形态，最大程度复用现有 Server 实现。
"""
from agent.mcp_client.client import DEVICE_SERVER_PATH, McpToolClient
from agent.services.device_registry import require_default_device


class DeviceMcpClient:
    """设备运行状态查询的 MCP 客户端门面。"""

    def __init__(self, server_path: str | None = None, timeout_seconds: float | None = None):
        self._client = McpToolClient(
            server_path=server_path or DEVICE_SERVER_PATH,
            timeout_seconds=timeout_seconds,
        )

    def query_status(self, user_id: str) -> str:
        """查询指定用户默认设备的运行状态，返回 MCP 工具文本结果。"""
        require_default_device(user_id)
        return self._client.call("query_device_status", {"user_id": str(user_id).strip()})


__all__ = ["DeviceMcpClient"]
