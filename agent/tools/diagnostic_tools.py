"""诊断 Agent 专用工具。

与对话 Agent 的 agent_tools.py 保持一致：统一用 _safe_call 包装异常。
所有重型依赖（知识库 Agent、设备状态服务）均懒加载，导入本模块不会触发
Embedding 模型或向量库的初始化。
"""
import traceback

from langchain_core.tools import tool
from utils.logger_handler import logger


def _safe_call(tool_name: str, func, *args, **kwargs) -> str:
    try:
        result = func(*args, **kwargs)
        logger.info({
            "event": "diagnostic_tool_success",
            "tool": tool_name,
            "args": args,
            "kwargs": kwargs,
        })
        return result
    except Exception as e:
        logger.error({
            "event": "diagnostic_tool_error",
            "tool": tool_name,
            "args": args,
            "kwargs": kwargs,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
        })
        return f"工具{tool_name}调用失败：{str(e)}，请稍后重试"


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


# ----------------------------------------------------------- 工具定义

@tool(description="查询指定用户设备的运行状态（覆盖率、清洁效率、耗材状态），入参为 user_id（数字字符串）")
def query_device_status(user_id: str) -> str:
    return _safe_call("query_device_status", _get_device_status_service().get_status, user_id)


@tool(description="根据错误码查询知识库故障排除手册，获取故障说明与处理建议，入参为 error_code（如 E01）")
def query_error_code(error_code: str) -> str:
    from agent.knowledge_agent import get_knowledge_agent
    agent = get_knowledge_agent()
    return _safe_call("query_error_code", agent.search_troubleshooting, error_code)


@tool(description="查询维护保养建议，入参为 query（维护相关检索词），检索范围限定在维护保养手册")
def query_maintenance(query: str) -> str:
    from agent.knowledge_agent import get_knowledge_agent
    agent = get_knowledge_agent()
    return _safe_call("query_maintenance", agent.search_maintenance, query)


@tool(description="全库知识检索，入参为 query，用于在没有限定手册时获取通用故障处理资料")
def retrieve_knowledge(query: str) -> str:
    from agent.knowledge_agent import get_knowledge_agent
    agent = get_knowledge_agent()
    return _safe_call("retrieve_knowledge", agent.retrieve, query)


def current_user_id() -> str:
    """诊断 Agent 执行设备状态类步骤时，自动获取当前用户 ID（mock 环境）。"""
    return _get_user_id_service().get_user_id()
