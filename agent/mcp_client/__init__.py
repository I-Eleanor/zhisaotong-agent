"""MCP Client 适配层：把 stdio MCP Server 的工具能力封装为同步业务接口。

结构：
- protocol.py      客户端 / 服务端共享的安全哨兵文本
- exceptions.py    MCP 连接 / 工具 / 超时异常（均属 MCPConnectionError 族）
- client.py        McpToolClient：stdio 连接 + 握手 + call_tool 的同步门面
- device_client.py DeviceMcpClient：query_status(user_id)
- log_client.py    LogMcpClient：query_logs(user_id, days)
"""
from agent.mcp_client.client import McpToolClient
from agent.mcp_client.device_client import DeviceMcpClient
from agent.mcp_client.exceptions import McpServerStartupError, McpTimeoutError, McpToolCallError
from agent.mcp_client.log_client import LogMcpClient
from agent.mcp_client.protocol import (
    LOGS_UNAVAILABLE_TEXT,
    SENTINEL_UNAVAILABLE_TEXTS,
    STATUS_UNAVAILABLE_TEXT,
)

__all__ = [
    "McpToolClient",
    "DeviceMcpClient",
    "LogMcpClient",
    "McpServerStartupError",
    "McpToolCallError",
    "McpTimeoutError",
    "STATUS_UNAVAILABLE_TEXT",
    "LOGS_UNAVAILABLE_TEXT",
    "SENTINEL_UNAVAILABLE_TEXTS",
]
