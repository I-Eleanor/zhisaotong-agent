"""诊断 Agent 专用工具。

与对话 Agent 的 agent_tools.py 保持一致：统一用 _safe_call 包装异常。
所有重型依赖（知识库 Agent、设备数据 Provider）均懒加载，导入本模块不会触发
Embedding 模型、向量库的初始化或 MCP 子进程启动。

设备状态 / 日志工具不再直连 CsvDeviceStatusService / MockDeviceLogService，
而是统一经 DeviceDataProvider 抽象（默认 MCP Provider，经 stdio 子进程
调用 device_server / log_server；可配置 DEVICE_DATA_PROVIDER=direct 直连）。
"""
from collections.abc import Callable

from langchain_core.tools import tool

from utils.logger_handler import log_safe_value, logger, safe_exception_fields

# 工具失败的固定安全文案（与 agent_tools.py 保持相同语义）：
# 不含异常类型 / 路径 / 密钥 / 服务响应 / 用户输入
_TOOL_FAILED_MESSAGE = "工具调用失败，请稍后重试"


def _safe_call(tool_name: str, func: Callable[..., str], *args: object, **kwargs: object) -> str:
    try:
        result = func(*args, **kwargs)
        logger.info({
            "event": "diagnostic_tool_success",
            "tool": tool_name,
            "args": log_safe_value(args),
            "kwargs": log_safe_value(kwargs),
        })
        return result
    except Exception as e:
        logger.error({
            "event": "diagnostic_tool_error",
            "tool": tool_name,
            "args": log_safe_value(args),
            "kwargs": log_safe_value(kwargs),
            **safe_exception_fields(e),
        })
        return _TOOL_FAILED_MESSAGE


# ----------------------------------------------------------- 懒加载单例

_device_data_provider = None


def _get_device_data_provider():
    """设备数据 Provider 懒加载单例；默认经 MCP Client 调用 MCP Server。

    模式由环境变量 DEVICE_DATA_PROVIDER 控制（mcp 默认 / direct），
    单元测试可通过 reset_device_data_provider() + 环境变量注入 direct 实现。
    """
    global _device_data_provider
    if _device_data_provider is None:
        from agent.services.device_data_provider import create_device_data_provider
        _device_data_provider = create_device_data_provider()
    return _device_data_provider


def reset_device_data_provider() -> None:
    """重置 Provider 单例（配置变更 / 测试隔离用）。"""
    global _device_data_provider
    _device_data_provider = None


# ----------------------------------------------------------- 原始调用（异常上抛）

def build_raw_knowledge_tools(knowledge_agent=None):
    """构造知识库类原始工具；knowledge_agent 可注入（应用容器/测试）。

    未注入时回退全局懒加载单例（旧行为不变）。
    """
    kb = knowledge_agent

    def _target():
        from agent.knowledge_agent import get_knowledge_agent
        return kb if kb is not None else get_knowledge_agent()

    def raw_query_error_code(error_code: str) -> str:
        """根据错误码查故障排除手册；异常直接抛出，供 ToolRouter 区分成功/失败。"""
        return str(_target().search_troubleshooting_strict(error_code))

    def raw_query_maintenance(query: str) -> str:
        return str(_target().search_maintenance_strict(query))

    def raw_retrieve_knowledge(query: str) -> str:
        return str(_target().retrieve_strict(query))

    return raw_query_error_code, raw_query_maintenance, raw_retrieve_knowledge


def raw_query_device_status(user_id: str) -> str:
    """查询设备运行状态；异常直接抛出，供 ToolRouter 区分成功/失败。

    执行链路（默认）：Provider(MCP) → stdio device_server → CSV 状态服务。
    """
    return str(_get_device_data_provider().query_status(user_id))


def raw_query_device_logs(user_id: str, days: int | str = 7) -> str:
    """查询设备运行日志；异常直接抛出，供 ToolRouter 区分成功/失败。

    days 由调用方归一化到 1-30；执行链路（默认）：
    Provider(MCP) → stdio log_server → Mock 日志服务。
    """
    from agent.mcp_client.log_client import normalize_days

    return str(_get_device_data_provider().query_logs(user_id, normalize_days(days)))


def raw_query_error_code(error_code: str) -> str:
    from agent.knowledge_agent import get_knowledge_agent
    return get_knowledge_agent().search_troubleshooting_strict(error_code)


def raw_query_maintenance(query: str) -> str:
    from agent.knowledge_agent import get_knowledge_agent
    return get_knowledge_agent().search_maintenance_strict(query)


def raw_retrieve_knowledge(query: str) -> str:
    from agent.knowledge_agent import get_knowledge_agent
    return get_knowledge_agent().retrieve_strict(query)


# ----------------------------------------------------------- 工具定义（LangChain @tool，异常兜底为安全提示）

@tool(description="查询指定用户设备的运行状态（覆盖率、清洁效率、耗材状态），入参为 user_id（数字字符串）")
def query_device_status(user_id: str) -> str:
    return _safe_call("query_device_status", raw_query_device_status, user_id)


@tool(description="查询指定用户设备最近 days 天的运行日志（含错误码、告警、运行时长），入参为 user_id（数字字符串）与 days（1-30，默认 7）")
def query_device_logs(user_id: str, days: int = 7) -> str:
    return _safe_call("query_device_logs", raw_query_device_logs, user_id, days)


@tool(description="根据错误码查询知识库故障排除手册，获取故障说明与处理建议，入参为 error_code（如 E01）")
def query_error_code(error_code: str) -> str:
    return _safe_call("query_error_code", raw_query_error_code, error_code)


@tool(description="查询维护保养建议，入参为 query（维护相关检索词），检索范围限定在维护保养手册")
def query_maintenance(query: str) -> str:
    return _safe_call("query_maintenance", raw_query_maintenance, query)


@tool(description="全库知识检索，入参为 query，用于在没有限定手册时获取通用故障处理资料")
def retrieve_knowledge(query: str) -> str:
    return _safe_call("retrieve_knowledge", raw_retrieve_knowledge, query)


def current_user_id() -> str:
    """诊断 Agent 执行设备类步骤时，自动获取当前用户 ID。

    优先取请求上下文中的 user_id（API 层写入，未传时为默认演示用户 1001）；
    不再随机生成，保证同一会话内设备归属稳定。
    """
    from agent.services.device_registry import DEFAULT_DEMO_USER_ID
    from utils.request_context import get_request_user_id

    return get_request_user_id() or DEFAULT_DEMO_USER_ID
