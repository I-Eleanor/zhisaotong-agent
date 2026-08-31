"""设备运行日志查询 MCP Server（stdio 传输）。

通过标准 MCP 协议把「设备运行日志查询」能力暴露给 Agent 或任意 MCP Host。
数据来源为 MockDeviceLogService：基于用户 ID 派生可复现的模拟日志
（含错误码、告警、运行时长等，格式 {timestamp, level, event, device_id, message}）。

查询链路（体现 用户 → 设备 → 日志 的归属关系）：
    user_id → 设备注册表校验归属（device_registry） → Mock 日志服务 → 日志文本

运行：python mcp_server/log_server.py
"""
import os
import sys

# 保证工程根在 sys.path，使 agent / utils 可导入（无论从哪个 cwd 启动）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from agent.mcp_client.log_client import normalize_days  # noqa: E402
from agent.mcp_client.protocol import LOGS_UNAVAILABLE_TEXT  # noqa: E402
from agent.services import create_device_log_service  # noqa: E402
from agent.services.device_registry import (  # noqa: E402
    device_status_header,
    normalize_user_id,
    require_default_device,
)
from utils.exceptions import ServiceUnavailableError  # noqa: E402
from utils.logger_handler import logger, safe_exception_fields  # noqa: E402

mcp = FastMCP("device-log-server")


@mcp.tool()
def query_device_logs(user_id: str, days: int = 7) -> str:
    """查询指定用户设备最近 days 天的运行日志（含错误码、告警、运行时长等）。

    入参 user_id 为数字字符串；days 为 1-30 的整数，默认 7。
    内部先校验用户-设备归属（未知用户视为数据不可用），再查询 Mock 日志服务。
    任何失败（归属校验、服务构造、底层异常）都只返回固定安全文本：
    异常原文可能含路径 / 密钥等内部信息，只进脱敏日志。
    """
    try:
        uid = normalize_user_id(user_id)
        device_id = require_default_device(uid)
        svc = create_device_log_service("mock")
        logs = svc.get_logs(uid, normalize_days(days))
        return f"{device_status_header(device_id)}\n{logs}"
    except ServiceUnavailableError as e:
        logger.warning({
            "event": "mcp_device_logs_unavailable",
            "error_type": type(e).__name__,
            "error_code": e.error_code,
        })
        return LOGS_UNAVAILABLE_TEXT
    except Exception as e:
        logger.error({
            "event": "mcp_device_logs_error",
            "error_type": type(e).__name__,
            **safe_exception_fields(e),
        })
        return LOGS_UNAVAILABLE_TEXT


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
