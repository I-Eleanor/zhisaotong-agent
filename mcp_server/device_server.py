"""设备状态查询 MCP Server（stdio 传输）。

通过标准 MCP 协议把「设备运行状态查询」能力暴露给 Agent 或任意 MCP Host。
数据来源复用项目内的 CsvDeviceStatusService（从 CSV 读取模拟设备数据）。

查询链路（体现 用户 → 设备 → 状态 的归属关系）：
    user_id → 设备注册表校验归属（device_registry） → CSV 状态服务 → 设备状态文本

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

from agent.mcp_client.protocol import STATUS_UNAVAILABLE_TEXT  # noqa: E402
from agent.services import create_device_status_service, create_user_id_service  # noqa: E402
from agent.services.device_registry import (  # noqa: E402
    device_status_header,
    normalize_user_id,
    require_default_device,
)
from utils.exceptions import ServiceUnavailableError  # noqa: E402
from utils.logger_handler import log_safe_text, logger, safe_exception_fields  # noqa: E402

mcp = FastMCP("device-status-server")


@mcp.tool()
def query_device_status(user_id: str) -> str:
    """查询指定用户设备的运行状态（覆盖率、清洁效率、耗材状态）。

    入参 user_id 为数字字符串（如 '1001'）；为空时使用默认演示用户。
    内部先校验用户-设备归属（未知用户视为数据不可用），再查询 CSV 状态服务。
    任何失败（归属校验、服务构造、数据缺失、底层异常）都只返回固定安全文本：
    异常原文可能含用户 ID / 路径 / 密钥等内部信息，只进脱敏日志。
    """
    try:
        uid = normalize_user_id(user_id)
        device_id = require_default_device(uid)
        svc = create_device_status_service("csv")
        status = svc.get_status(uid)
        return f"{device_status_header(device_id)}\n{status}"
    except ServiceUnavailableError as e:
        logger.warning({
            "event": "mcp_device_status_unavailable",
            "user_id": log_safe_text(uid) if user_id else "",
            "error_type": type(e).__name__,
            "error_code": e.error_code,
            "error_msg": log_safe_text(str(e)),
        })
        return STATUS_UNAVAILABLE_TEXT
    except Exception as e:
        logger.error({
            "event": "mcp_device_status_error",
            "error_type": type(e).__name__,
            **safe_exception_fields(e),
        })
        return STATUS_UNAVAILABLE_TEXT


@mcp.tool()
def query_current_user() -> str:
    """获取当前默认用户 ID（mock 环境，演示用）。"""
    return create_user_id_service("mock").get_user_id()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
