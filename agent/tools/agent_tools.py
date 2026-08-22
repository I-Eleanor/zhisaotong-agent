from datetime import datetime

from langchain_core.tools import tool

from agent.services import (
    create_external_data_service,
    create_user_id_service,
    create_weather_service,
)
from utils.logger_handler import log_safe_value, logger, safe_exception_fields

weather_service = create_weather_service("mock")
user_id_service = create_user_id_service("mock")
external_data_service = create_external_data_service("csv")

# 工具失败的固定安全文案：不含异常类型 / 路径 / 密钥 / 服务响应 / 用户输入
_TOOL_FAILED_MESSAGE = "工具调用失败，请稍后重试"


def _safe_call(tool_name: str, func, *args, **kwargs) -> str:
    try:
        result = str(func(*args, **kwargs))
        logger.info({
            "event": "tool_success",
            "tool": tool_name,
            "args": log_safe_value(args),
            "kwargs": log_safe_value(kwargs),
        })
        return result
    except Exception as e:
        logger.error({
            "event": "tool_error",
            "tool": tool_name,
            "args": log_safe_value(args),
            "kwargs": log_safe_value(kwargs),
            **safe_exception_fields(e),
        })
        return _TOOL_FAILED_MESSAGE


def build_rag_tool(knowledge_agent=None):
    """构造 rag_summarize 工具；knowledge_agent 可注入（应用容器/测试）。

    未注入时回退全局懒加载单例（旧行为不变），重型依赖仍延迟到首次调用。
    """
    kb = knowledge_agent

    @tool(description="从向量存储中检索参考资料")
    def rag_summarize(query: str) -> str:
        from agent.knowledge_agent import get_knowledge_agent
        target = kb if kb is not None else get_knowledge_agent()
        return _safe_call("rag_summarize", target.retrieve, query)

    return rag_summarize


def build_conversation_tools(knowledge_agent=None) -> list:
    """构造对话 Agent 全套工具；RAG 工具绑定注入的知识库 Agent。"""
    return [
        build_rag_tool(knowledge_agent),
        get_weather,
        get_user_id,
        get_current_month,
        fetch_external_data,
        fill_context_for_report,
    ]


# 模块级默认工具（未注入依赖时的回退，保持旧调用方兼容）
rag_summarize = build_rag_tool()


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
