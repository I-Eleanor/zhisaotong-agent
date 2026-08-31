"""设备日志 MCP Client：user_id → 默认 device_id → log_server.query_device_logs。

与设备状态 Client 相同的归属校验策略；days 限制在 1-30。
"""
from agent.mcp_client.client import LOG_SERVER_PATH, McpToolClient
from agent.services.device_registry import require_default_device

MIN_LOG_DAYS = 1
MAX_LOG_DAYS = 30
DEFAULT_LOG_DAYS = 7


def normalize_days(days: int | str | float | None) -> int:
    """把任意输入归一化为 [1, 30] 内的整数天数；非法值回退默认 7 天。"""
    try:
        value = int(days)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_LOG_DAYS
    return max(MIN_LOG_DAYS, min(value, MAX_LOG_DAYS))


class LogMcpClient:
    """设备运行日志查询的 MCP 客户端门面。"""

    def __init__(self, server_path: str | None = None, timeout_seconds: float | None = None):
        self._client = McpToolClient(
            server_path=server_path or LOG_SERVER_PATH,
            timeout_seconds=timeout_seconds,
        )

    def query_logs(self, user_id: str, days: int = 7) -> str:
        """查询指定用户默认设备最近 days 天的运行日志。"""
        require_default_device(user_id)
        safe_days = normalize_days(days)
        return self._client.call(
            "query_device_logs",
            {"user_id": str(user_id).strip(), "days": safe_days},
        )


__all__ = ["LogMcpClient", "normalize_days", "MIN_LOG_DAYS", "MAX_LOG_DAYS", "DEFAULT_LOG_DAYS"]
