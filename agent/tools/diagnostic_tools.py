"""诊断 Agent 专用工具。

与对话 Agent 的 agent_tools.py 保持一致：统一用 _safe_call 包装异常。
所有重型依赖（知识库 Agent、设备状态服务）均懒加载，导入本模块不会触发
Embedding 模型或向量库的初始化。
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

_status_service = None
_user_id_service = None


def _get_device_status_service():
    global _status_service
    if _status_service is None:
        from agent.services import create_device_status_service
        _status_service = create_device_status_service("csv")
    return _status_service


def _get_user_id_service():
    global _user_id_service
    if _user_id_service is None:
        from agent.services import create_user_id_service
        _user_id_service = create_user_id_service("mock")
    return _user_id_service


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
    """查询设备运行状态；异常直接抛出，供 ToolRouter 区分成功/失败。"""
    return str(_get_device_status_service().get_status(user_id))


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
    """诊断 Agent 执行设备状态类步骤时，自动获取当前用户 ID（mock 环境）。"""
    return str(_get_user_id_service().get_user_id())
