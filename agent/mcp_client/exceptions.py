"""MCP Client 侧异常体系。

全部继承 utils.exceptions.MCPConnectionError（stage="mcp"、
error_code=TOOL_UNAVAILABLE），ToolRouter 只需一处 except 即可把
任何 MCP 连接 / 调用 / 超时失败转换为安全降级步骤。

异常 message 只进服务端脱敏日志；客户端可见文案由 ToolRouter 的
固定模板给出，绝不拼接异常原文。
"""
from utils.exceptions import MCPConnectionError


class McpServerStartupError(MCPConnectionError):
    """MCP Server 子进程启动 / 握手失败（命令不存在、立即退出等）。"""


class McpToolCallError(MCPConnectionError):
    """MCP 工具调用失败：工具不存在、返回 isError、返回内容为空。"""


class McpTimeoutError(MCPConnectionError):
    """MCP 调用超时（含子进程启动与工具执行全程）。"""


__all__ = ["McpServerStartupError", "McpToolCallError", "McpTimeoutError"]
