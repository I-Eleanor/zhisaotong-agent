"""设备数据 Provider 抽象：统一接口 + Direct / MCP 两种实现。

统一接口（DeviceDataProvider）：
    query_status(user_id) -> str
    query_logs(user_id, days=7) -> str

两种实现：
- DirectDeviceDataProvider：进程内直连 CsvDeviceStatusService / MockDeviceLogService
  （保留项目的直接调用能力，供单元测试与 MCP 故障切换使用）；
- McpDeviceDataProvider：经 MCP Client（stdio 子进程）调用 device_server / log_server
  （主链路默认实现，数据服务边界可替换）。

选择方式：
- 环境变量 DEVICE_DATA_PROVIDER = "mcp"（默认）| "direct"；
- 或构造时直接注入 Provider 实例（ToolRouter / 测试）。

两种实现共享同一归属校验（device_registry）与同一输出格式
（设备ID 头 + 服务文本），行为完全一致。
"""
import os
from typing import Protocol

from agent.mcp_client.device_client import DeviceMcpClient
from agent.mcp_client.log_client import LogMcpClient, normalize_days
from agent.services.device_registry import device_status_header, require_default_device
from utils.logger_handler import log_safe_text, logger

PROVIDER_MODE_MCP = "mcp"
PROVIDER_MODE_DIRECT = "direct"
DEFAULT_PROVIDER_MODE = PROVIDER_MODE_MCP

# 环境变量名：切换设备数据 Provider（mcp / direct）
DEVICE_DATA_PROVIDER_ENV = "DEVICE_DATA_PROVIDER"


class DeviceDataProvider(Protocol):
    """设备数据 Provider 统一接口（Direct 与 MCP 实现的共同契约）。"""

    def query_status(self, user_id: str) -> str:
        """查询指定用户默认设备的运行状态。"""
        ...

    def query_logs(self, user_id: str, days: int = 7) -> str:
        """查询指定用户默认设备最近 days 天的运行日志。"""
        ...


class DirectDeviceDataProvider:
    """直连实现：进程内直接调用 CSV 状态服务 / Mock 日志服务。

    保留项目原有的直接调用能力；与 MCP 实现共享归属校验与输出格式，
    便于单元测试注入与 MCP 故障时快速切换。
    """

    def __init__(self, status_service_factory=None, log_service_factory=None):
        # 工厂可注入（测试替身）；默认走 agent.services 的标准工厂
        if status_service_factory is None:
            from agent.services import create_device_status_service
            status_service_factory = create_device_status_service
        if log_service_factory is None:
            from agent.services import create_device_log_service
            log_service_factory = create_device_log_service
        self._status_factory = status_service_factory
        self._log_factory = log_service_factory

    def query_status(self, user_id: str) -> str:
        device_id = require_default_device(user_id)
        status = self._status_factory("csv").get_status(user_id)
        return f"{device_status_header(device_id)}\n{status}"

    def query_logs(self, user_id: str, days: int = 7) -> str:
        device_id = require_default_device(user_id)
        logs = self._log_factory("mock").get_logs(user_id, normalize_days(days))
        return f"{device_status_header(device_id)}\n{logs}"


class McpDeviceDataProvider:
    """MCP 实现：经 MCP Client 子进程调用 device_server / log_server。

    委托前先做一次归属校验（纵深防御）：未知用户不发起任何子进程调用，
    Client 内部会再校验一次，保证"用户不能访问他人设备"在两层都成立。
    """

    def __init__(self, device_client: DeviceMcpClient | None = None,
                 log_client: LogMcpClient | None = None):
        self.device_client = device_client or DeviceMcpClient()
        self.log_client = log_client or LogMcpClient()

    def query_status(self, user_id: str) -> str:
        require_default_device(user_id)
        return self.device_client.query_status(user_id)

    def query_logs(self, user_id: str, days: int = 7) -> str:
        require_default_device(user_id)
        return self.log_client.query_logs(user_id, days)


def resolve_provider_mode(mode: str | None = None) -> str:
    """解析 Provider 模式：显式参数 > 环境变量 > 默认 mcp；未知值回退 mcp。"""
    value = (mode or os.getenv(DEVICE_DATA_PROVIDER_ENV, "") or "").strip().lower()
    if value == PROVIDER_MODE_DIRECT:
        return PROVIDER_MODE_DIRECT
    if value != PROVIDER_MODE_MCP and value:
        logger.warning({
            "event": "device_provider_mode_invalid",
            "mode": log_safe_text(value),
            "fallback": DEFAULT_PROVIDER_MODE,
        })
    return DEFAULT_PROVIDER_MODE


def create_device_data_provider(mode: str | None = None) -> DeviceDataProvider:
    """按模式构造设备数据 Provider；默认 mcp（主链路经 MCP Client 执行）。"""
    resolved = resolve_provider_mode(mode)
    logger.info({"event": "device_provider_init", "mode": resolved})
    if resolved == PROVIDER_MODE_DIRECT:
        return DirectDeviceDataProvider()
    return McpDeviceDataProvider()


__all__ = [
    "DeviceDataProvider",
    "DirectDeviceDataProvider",
    "McpDeviceDataProvider",
    "create_device_data_provider",
    "resolve_provider_mode",
    "DEVICE_DATA_PROVIDER_ENV",
]
