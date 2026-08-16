import traceback
from datetime import datetime

from langchain_core.tools import tool

from agent.services import (
    create_external_data_service,
    create_user_id_service,
    create_weather_service,
)
from utils.logger_handler import logger


def _get_rag_summarize(query: str) -> str:
    """懒加载知识库 Agent 并检索总结，避免导入本模块即触发 Embedding/向量库初始化。

    与 diagnostic_tools.py 保持一致：重型依赖全部延迟到首次调用才创建。
    """
    from agent.knowledge_agent import get_knowledge_agent
    return get_knowledge_agent().retrieve(query)


weather_service = create_weather_service("mock")
user_id_service = create_user_id_service("mock")
external_data_service = create_external_data_service("csv")


def _safe_call(tool_name: str, func, *args, **kwargs) -> str:
    try:
        result = func(*args, **kwargs)
        logger.info({
            "event": "tool_success",
            "tool": tool_name,
            "args": args,
            "kwargs": kwargs,
        })
        return result
    except Exception as e:
        logger.error({
            "event": "tool_error",
            "tool": tool_name,
            "args": args,
            "kwargs": kwargs,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
        })
        return f"工具{tool_name}调用失败：{str(e)}，请稍后重试"


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return _safe_call("rag_summarize", _get_rag_summarize, query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return _safe_call("get_weather", weather_service.get_weather, city)


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return _safe_call("get_user_id", user_id_service.get_user_id)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return _safe_call("get_current_month", lambda: datetime.now().strftime("%Y-%m"))


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    return _safe_call("fetch_external_data", external_data_service.fetch_data, user_id, month)


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
