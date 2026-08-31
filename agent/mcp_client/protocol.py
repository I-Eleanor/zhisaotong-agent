"""MCP 客户端 / 服务端共享的安全协议常量。

这些固定文案是 MCP Server 与 MCP Client 之间显式约定的"哨兵文本"：
Server 在数据不可用时返回它们（而不是抛异常，避免向任意 MCP Host 泄漏
内部细节），Client 识别后还原为 ServiceUnavailableError，保证
"数据不可用 → StepResult(success=False)" 的语义跨 MCP 边界不丢失。

仅放常量，不引入任何依赖，Server 与 Client 均可安全导入。
"""

# device_server.query_device_status 数据不可用时的固定安全文案
STATUS_UNAVAILABLE_TEXT = "设备状态数据暂时不可用，请稍后重试。"

# log_server.query_device_logs 数据不可用时的固定安全文案
LOGS_UNAVAILABLE_TEXT = "设备日志数据暂时不可用，请稍后重试。"

# 所有哨兵文本（Client 侧统一识别）
SENTINEL_UNAVAILABLE_TEXTS: frozenset[str] = frozenset({
    STATUS_UNAVAILABLE_TEXT,
    LOGS_UNAVAILABLE_TEXT,
})
