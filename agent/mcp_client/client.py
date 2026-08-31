"""MCP stdio 客户端基础封装（同步门面）。

职责（对 scripts/mcp_smoke.py 的业务化封装，而非简单复制测试函数）：
- 使用 StdioServerParameters 配置 Server 并以 sys.executable 拉起子进程；
- 建立 stdio_client + ClientSession 连接并完成 initialize() 握手；
- call_tool() 调用工具并提取首个文本内容；
- 全程超时控制（默认 30s，覆盖 启动→握手→调用→关闭）；
- 连接失败 / 工具不存在 / 超时 / 空结果统一转 MCPConnectionError 族异常；
- 识别 Server 返回的"数据不可用"哨兵文本并还原为 ServiceUnavailableError；
- 每次调用结束关闭会话与子进程（无长连接、无泄漏）。

同步语义：ToolRouter / Agent 链路是同步的，内部用 asyncio.run() 独立事件
循环执行一次完整的"连接→调用→关闭"生命周期；只能在无运行中事件循环的
线程调用（SSE 桥生产线程、FastAPI 线程池均满足）。
"""
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import TextContent

from agent.mcp_client.exceptions import McpServerStartupError, McpTimeoutError, McpToolCallError
from agent.mcp_client.protocol import SENTINEL_UNAVAILABLE_TEXTS
from utils.exceptions import ServiceUnavailableError
from utils.logger_handler import log_safe_text, logger, safe_exception_fields

# 单次 MCP 调用（含子进程启动 + 握手 + 工具执行 + 关闭）的默认超时（秒）
DEFAULT_CALL_TIMEOUT_SECONDS = 30.0

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEVICE_SERVER_PATH = os.path.join(_PROJECT_ROOT, "mcp_server", "device_server.py")
LOG_SERVER_PATH = os.path.join(_PROJECT_ROOT, "mcp_server", "log_server.py")


def default_call_timeout() -> float:
    """从环境变量读取超时（MCP_CALL_TIMEOUT_SECONDS），非法值回退默认。"""
    raw = os.getenv("MCP_CALL_TIMEOUT_SECONDS", "")
    try:
        value = float(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    return DEFAULT_CALL_TIMEOUT_SECONDS


class McpToolClient:
    """面向单个 MCP Server 的同步工具调用客户端（每次调用一个完整生命周期）。"""

    def __init__(self, server_path: str, timeout_seconds: float | None = None):
        self.server_path = server_path
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else default_call_timeout()
        )

    # --------------------------------------------------------------- 对外入口

    def call(self, tool_name: str, arguments: dict | None = None) -> str:
        """同步调用 MCP 工具并返回文本结果。

        失败语义：
        - ServiceUnavailableError：Server 明确返回"数据不可用"哨兵文本；
        - McpServerStartupError / McpToolCallError / McpTimeoutError：
          连接 / 工具 / 超时类失败（均属 MCPConnectionError 族）。
        """
        args = {k: v for k, v in (arguments or {}).items() if v is not None}
        logger.info({
            "event": "mcp_client_call",
            "tool": tool_name,
            "timeout_s": self.timeout_seconds,
        })
        # 同步门面依赖 asyncio.run() 建立独立事件循环：只能在无运行中循环的
        # 线程调用（SSE 桥生产线程 / FastAPI 线程池均满足）。若调用方误在
        # 异步上下文直接调用，这里显式降级为 MCP 连接失败，不抛 RuntimeError。
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            logger.warning({"event": "mcp_client_sync_in_async_context", "tool": tool_name})
            raise McpServerStartupError("MCP Client 不支持在异步上下文中同步调用")
        try:
            return asyncio.run(self._call_async(tool_name, args))
        except (ServiceUnavailableError, McpServerStartupError, McpToolCallError, McpTimeoutError):
            raise
        except TimeoutError as e:
            raise McpTimeoutError("MCP 调用超时") from e
        except Exception as e:
            # anyio / 子进程 / 协议层异常：统一归并为连接类失败，细节只进日志
            logger.warning({
                "event": "mcp_client_unexpected_error",
                "tool": tool_name,
                **safe_exception_fields(e),
            })
            raise McpServerStartupError("MCP 连接失败") from e

    async def _call_async(self, tool_name: str, arguments: dict) -> str:
        try:
            return await asyncio.wait_for(
                self._call_session(tool_name, arguments),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as e:
            raise McpTimeoutError("MCP 调用超时") from e

    async def _call_session(self, tool_name: str, arguments: dict) -> str:
        params = StdioServerParameters(command=sys.executable, args=[self.server_path])
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
        except McpError as e:
            # 协议层错误（含工具不存在 / 参数校验失败）：归并为工具调用失败
            logger.warning({
                "event": "mcp_client_protocol_error",
                "tool": tool_name,
                **safe_exception_fields(e),
            })
            raise McpToolCallError("MCP 工具调用失败") from e
        except McpToolCallError:
            raise
        except Exception as e:
            logger.warning({
                "event": "mcp_client_connect_failed",
                "tool": tool_name,
                **safe_exception_fields(e),
            })
            raise McpServerStartupError("MCP Server 连接失败") from e

        return self._extract_text(tool_name, result)

    # --------------------------------------------------------------- 结果解析

    def _extract_text(self, tool_name: str, result) -> str:
        """从 CallToolResult 提取文本并映射失败语义；不向客户端泄漏异常原文。"""
        if getattr(result, "isError", False):
            logger.warning({"event": "mcp_client_tool_error_result", "tool": tool_name})
            raise McpToolCallError("MCP 工具返回错误结果")

        texts = [
            item.text
            for item in (getattr(result, "content", None) or [])
            if isinstance(item, TextContent) and getattr(item, "text", None)
        ]
        text = "\n".join(texts).strip()

        if not text:
            logger.warning({"event": "mcp_client_empty_result", "tool": tool_name})
            raise McpToolCallError("MCP 工具返回内容为空")

        # 哨兵文本 → 服务不可用（保持与直连服务一致的失败语义）
        if text in SENTINEL_UNAVAILABLE_TEXTS:
            logger.info({
                "event": "mcp_client_service_unavailable",
                "tool": tool_name,
                "text": log_safe_text(text),
            })
            raise ServiceUnavailableError("MCP 工具报告设备数据不可用")

        return text


__all__ = ["McpToolClient", "DEVICE_SERVER_PATH", "LOG_SERVER_PATH", "default_call_timeout"]
