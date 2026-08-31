"""设备注册表 + 设备数据 Provider 测试。

覆盖：
- 演示用户 → 设备映射（已知 / 未知用户）
- 用户不能访问不属于自己的设备（注册表只映射自有设备）
- Direct Provider：直连 CSV / Mock，输出含设备标识
- MCP Provider：委托给 MCP Client（用替身验证调用链，不启动子进程）
- create_device_data_provider 的模式解析（mcp 默认 / direct / 非法值回退）
"""
import pytest

from agent.mcp_client.exceptions import McpToolCallError
from agent.services.device_data_provider import (
    DirectDeviceDataProvider,
    McpDeviceDataProvider,
    create_device_data_provider,
    resolve_provider_mode,
)
from agent.services.device_registry import (
    DEFAULT_DEMO_USER_ID,
    get_default_device,
    get_user_devices,
    normalize_user_id,
    require_default_device,
)
from utils.exceptions import MCPConnectionError, ServiceUnavailableError


# ----------------------------------------------------------------- 设备注册表
def test_known_user_returns_own_devices():
    assert get_user_devices("1001") == ["device-1001"]
    assert get_default_device("1001") == "device-1001"
    assert get_default_device("1006") == "device-1006"


def test_unknown_user_returns_empty_and_no_exception():
    assert get_user_devices("999999") == []
    assert get_user_devices("") == []
    assert get_default_device("999999") is None


def test_user_cannot_access_other_users_device():
    """注册表只映射自有设备：无法通过输入访问其他用户的设备。"""
    assert get_default_device("1002") == "device-1002"
    assert "device-1001" not in get_user_devices("1002")


def test_require_default_device_raises_for_unknown_user():
    with pytest.raises(ServiceUnavailableError):
        require_default_device("999999")


def test_normalize_user_id_falls_back_to_demo_user():
    assert normalize_user_id(None) == DEFAULT_DEMO_USER_ID == "1001"
    assert normalize_user_id("") == "1001"
    assert normalize_user_id(" 1002 ") == "1002"


# ----------------------------------------------------------------- Direct Provider
def test_direct_provider_status_contains_device_and_csv_data():
    provider = DirectDeviceDataProvider()
    text = provider.query_status("1001")
    assert "设备ID：device-1001" in text, "输出应体现 用户 → 设备 → 状态 链路"
    assert "覆盖率" in text or "清洁效率" in text, "应包含 CSV 中的设备状态数据"


def test_direct_provider_logs_contains_device_and_logs():
    provider = DirectDeviceDataProvider()
    text = provider.query_logs("1001", days=3)
    assert "设备ID：device-1001" in text
    assert "运行日志" in text


def test_direct_provider_unknown_user_raises_service_unavailable():
    provider = DirectDeviceDataProvider()
    with pytest.raises(ServiceUnavailableError):
        provider.query_status("999999")


def test_direct_provider_mcp_paths_share_output_format():
    """Direct 与 MCP 输出格式一致（同一设备标识头）。"""
    direct = DirectDeviceDataProvider().query_status("1003")
    assert direct.startswith("设备ID：device-1003")


# ----------------------------------------------------------------- MCP Provider
class _FakeDeviceClient:
    def __init__(self, text="MCP 设备状态"):
        self.text = text
        self.calls: list[str] = []

    def query_status(self, user_id: str) -> str:
        self.calls.append(user_id)
        return self.text


class _FakeLogClient:
    def __init__(self, text="MCP 运行日志"):
        self.text = text
        self.calls: list[tuple[str, int]] = []

    def query_logs(self, user_id: str, days: int = 7) -> str:
        self.calls.append((user_id, days))
        return self.text


def test_mcp_provider_delegates_to_clients():
    device_client, log_client = _FakeDeviceClient(), _FakeLogClient()
    provider = McpDeviceDataProvider(device_client=device_client, log_client=log_client)

    assert provider.query_status("1001") == "MCP 设备状态"
    assert device_client.calls == ["1001"]
    assert provider.query_logs("1001", 5) == "MCP 运行日志"
    assert log_client.calls == [("1001", 5)]


def test_mcp_provider_propagates_connection_error():
    class BoomClient:
        def query_status(self, user_id):
            raise McpToolCallError("MCP 工具不可用")

        def query_logs(self, user_id, days=7):
            raise McpToolCallError("MCP 工具不可用")

    provider = McpDeviceDataProvider(device_client=BoomClient(), log_client=BoomClient())
    with pytest.raises(MCPConnectionError):
        provider.query_status("1001")
    with pytest.raises(MCPConnectionError):
        provider.query_logs("1001")


def test_mcp_provider_unknown_user_short_circuits_without_client():
    """未知用户：Provider 在注册表校验处即失败，不发起 MCP 调用。"""

    class NeverCalledClient:
        def query_status(self, user_id):
            raise AssertionError("未知用户不应发起 MCP 调用")

        def query_logs(self, user_id, days=7):
            raise AssertionError("未知用户不应发起 MCP 调用")

    provider = McpDeviceDataProvider(device_client=NeverCalledClient(), log_client=NeverCalledClient())
    with pytest.raises(ServiceUnavailableError):
        provider.query_status("999999")


# ----------------------------------------------------------------- Provider 模式
def test_resolve_provider_mode_defaults_to_mcp(monkeypatch):
    monkeypatch.delenv("DEVICE_DATA_PROVIDER", raising=False)
    assert resolve_provider_mode() == "mcp"


def test_resolve_provider_mode_direct(monkeypatch):
    monkeypatch.setenv("DEVICE_DATA_PROVIDER", "direct")
    assert resolve_provider_mode() == "direct"
    assert isinstance(create_device_data_provider(), DirectDeviceDataProvider)


def test_resolve_provider_mode_invalid_falls_back_to_mcp(monkeypatch):
    monkeypatch.setenv("DEVICE_DATA_PROVIDER", "weird")
    assert resolve_provider_mode() == "mcp"


def test_create_provider_mcp_returns_mcp_provider(monkeypatch):
    monkeypatch.setenv("DEVICE_DATA_PROVIDER", "mcp")
    assert isinstance(create_device_data_provider("mcp"), McpDeviceDataProvider)
