"""超时、重试与降级工具。

提供：
- with_timeout：给同步/异步调用加超时
- retry_with_backoff：指数退避重试（仅对可重试错误）
- MCP 降级策略
"""
import time
from collections.abc import Callable
from typing import Any

from utils.exceptions import AgentProjectError, ModelInvocationError
from utils.logger_handler import logger


def with_timeout(func: Callable, timeout_seconds: float, error_class: type = ModelInvocationError, label: str = "") -> Any:
    """给同步调用加超时（使用线程 + join 超时）。"""
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
    """指数退避重试，仅对可重试的 AgentProjectError 重试。

    不应重试的错误（retryable=False）直接抛出。
    非项目异常也不重试（如参数错误、JSON 解析失败）。
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except AgentProjectError as e:
            last_exc = e
            if not e.retryable:
                logger.warning({"event": "retry_skip_not_retryable", "label": label, "stage": e.stage, "error": str(e)})
                raise
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning({
                    "event": "retry_attempt",
                    "label": label,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay_s": delay,
                    "error": str(e),
                })
                time.sleep(delay)
            else:
                logger.error({"event": "retry_exhausted", "label": label, "attempts": max_retries + 1, "error": str(e)})
                raise
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning({
                    "event": "retry_attempt_unexpected",
                    "label": label,
                    "attempt": attempt + 1,
                    "delay_s": delay,
                    "error": str(e),
                })
                time.sleep(delay)
            else:
                raise
    raise last_exc


def mcp_fallback_response(tool_name: str, error: Exception) -> str:
    """MCP 不可用时的降级响应。"""
    logger.warning({
        "event": "mcp_fallback",
        "tool": tool_name,
        "error": str(error),
        "stage": "mcp",
    })
    return f"⚠️ 实时设备数据不可用（{tool_name}），以下建议基于静态知识库，可能不够准确。"
