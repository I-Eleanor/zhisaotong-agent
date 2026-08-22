"""超时、重试与降级工具。

提供：
- with_timeout：给同步/异步调用加超时
- retry_with_backoff：指数退避重试（retryable=False 的确定性项目异常不重试）
- MCP 降级策略
"""
import time
from collections.abc import Callable
from typing import Any

from utils.exceptions import AgentProjectError, ModelInvocationError
from utils.logger_handler import logger, safe_exception_fields


def with_timeout(func: Callable, timeout_seconds: float, error_class: type[AgentProjectError] = ModelInvocationError, label: str = "") -> Any:
    """给同步调用加超时（使用线程 + join 超时）。

    注意：超时只是放弃等待结果，后台线程仍会跑完（线程无法真正取消）；
    LLM / HTTP 调用应优先使用 SDK 原生超时（如 ChatOpenAI 的 request_timeout），
    本工具仅作为无原生超时能力的同步调用的兜底。
    """
    import threading

    result = [None]
    exc = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        raise error_class(
            f"{label or func.__name__} 超时（{timeout_seconds}s）",
            stage=error_class.stage,
            retryable=True,
        )

    if exc[0] is not None:
        raise exc[0]

    return result[0]


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    label: str = "",
) -> Any:
    """指数退避重试：可重试项目异常与普通异常均按退避重试。

    不重试的只有 retryable=False 的项目异常（工具执行失败、
    文档解析失败等确定性错误），直接抛出。
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except AgentProjectError as e:
            if not e.retryable:
                logger.warning({
                    "event": "retry_skip_not_retryable",
                    "label": label,
                    "stage": e.stage,
                    **safe_exception_fields(e),
                })
                raise
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning({
                    "event": "retry_attempt",
                    "label": label,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay_s": delay,
                    **safe_exception_fields(e),
                })
                time.sleep(delay)
            else:
                logger.error({
                    "event": "retry_exhausted",
                    "label": label,
                    "attempts": max_retries + 1,
                    **safe_exception_fields(e),
                })
                raise
        except Exception as e:
            # 普通异常：与可重试项目异常同等按指数退避重试（原始行为）——
            # 网络抖动 / 临时故障常以非项目异常形态上抛，直接放弃会损失可用性
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning({
                    "event": "retry_attempt_unexpected",
                    "label": label,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay_s": delay,
                    **safe_exception_fields(e),
                })
                time.sleep(delay)
            else:
                logger.error({
                    "event": "retry_exhausted",
                    "label": label,
                    "attempts": max_retries + 1,
                    **safe_exception_fields(e),
                })
                raise
    raise RuntimeError("unreachable: retry loop must return or raise")


def mcp_fallback_response(tool_name: str, error: Exception) -> str:
    """MCP 不可用时的降级响应。

    客户端可见文案只允许固定安全文本：tool_name 必须命中内部白名单
    （诊断工具 + MCP server 暴露的工具名），否则退化为不含工具名的通用文案，
    防止调用方把任意文本当作工具名拼进客户端响应。
    """
    safe_tool = tool_name if tool_name in _MCP_FALLBACK_TOOL_WHITELIST else ""
    logger.warning({
        "event": "mcp_fallback",
        "tool": tool_name if safe_tool else "<unknown>",
        "stage": "mcp",
        **safe_exception_fields(error),
    })
    if not safe_tool:
        return "⚠️ 实时设备数据不可用，以下建议基于静态知识库，可能不够准确。"
    return f"⚠️ 实时设备数据不可用（{safe_tool}），以下建议基于静态知识库，可能不够准确。"


# 客户端可见的工具名白名单：诊断 Agent 工具（agent/diagnostic/schemas.py 的
# ALLOWED_TOOLS）+ MCP server 暴露的工具名；utils 不反向依赖 agent 包，此处显式枚举
_MCP_FALLBACK_TOOL_WHITELIST = frozenset({
    "query_device_status",
    "query_error_code",
    "query_maintenance",
    "retrieve_knowledge",
    "get_device_status",
    "query_current_user",
})
