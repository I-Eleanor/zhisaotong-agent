"""演示用户与设备的静态映射（轻量身份与设备归属模型）。

设计边界：
- 仅用于演示：不连接真实设备、不实现数据库、不实现注册流程；
- 设备状态与运行日志均为可复现的模拟数据（CSV / Mock 数据源）；
- 每个演示用户只映射到自己的设备，用户输入无法访问不属于自己的设备；
- 未知用户返回空列表 / None（查询侧据此抛出安全错误，不泄露内部细节）。

设备 ID 与 MockDeviceLogService 生成的 device-{user_id} 标识保持一致。
"""
from utils.exceptions import ServiceUnavailableError
from utils.logger_handler import log_safe_text, logger

# 默认演示用户：请求未携带 user_id 时使用（保证同一用户稳定落到同一设备，
# 不再随机生成用户 ID）
DEFAULT_DEMO_USER_ID = "1001"

# 演示用户 → 绑定设备列表。覆盖 CSV 数据源中的全部演示用户（1001-1010），
# 每人仅映射自己的设备，天然隔离用户间的设备访问。
DEMO_USER_DEVICES: dict[str, list[str]] = {
    "1001": ["device-1001"],
    "1002": ["device-1002"],
    "1003": ["device-1003"],
    "1004": ["device-1004"],
    "1005": ["device-1005"],
    "1006": ["device-1006"],
    "1007": ["device-1007"],
    "1008": ["device-1008"],
    "1009": ["device-1009"],
    "1010": ["device-1010"],
}


def get_user_devices(user_id: str) -> list[str]:
    """返回该用户绑定的设备 ID 列表；未知用户返回空列表。"""
    if not user_id:
        return []
    devices = DEMO_USER_DEVICES.get(str(user_id).strip())
    return list(devices) if devices else []


def get_default_device(user_id: str) -> str | None:
    """返回该用户的默认设备 ID；未绑定设备时返回 None。"""
    devices = get_user_devices(user_id)
    return devices[0] if devices else None


def require_default_device(user_id: str) -> str:
    """返回该用户的默认设备 ID；未绑定设备时抛 ServiceUnavailableError。

    异常消息仅描述"未绑定设备"这一事实，不携带用户原始输入、路径等
    内部信息；调用方（MCP Server / Provider）据此返回固定安全文案。
    """
    device_id = get_default_device(user_id)
    if device_id is None:
        logger.warning({"event": "device_not_bound", "user_id": log_safe_text(user_id)})
        raise ServiceUnavailableError("当前用户未绑定演示设备，无法查询设备数据")
    return device_id


def device_status_header(device_id: str) -> str:
    """设备状态/日志输出的统一设备标识行（MCP Server 与 Direct Provider 共用）。"""
    return f"设备ID：{device_id}"


def normalize_user_id(user_id: str | None) -> str:
    """归一化用户输入的 user_id；为空时回退默认演示用户。"""
    value = (user_id or "").strip()
    return value or DEFAULT_DEMO_USER_ID
