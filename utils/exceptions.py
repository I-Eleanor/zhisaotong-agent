"""统一业务异常体系。

所有自定义异常继承自 AgentProjectError，按阶段分类：
- ConfigurationError：配置缺失或无效
- ModelInvocationError：LLM / Embedding 调用失败
- RetrievalError：向量检索失败
- ToolExecutionError：工具执行失败
- MCPConnectionError：MCP Server 连接失败
- DocumentParseError：文档解析失败
- StreamExecutionError：SSE 流式输出异常

每个异常携带：
- stage：出错阶段（config / model / retrieval / tool / mcp / document / stream）
- retryable：是否可重试
- original：原始异常（不暴露给前端）
"""


class AgentProjectError(Exception):
    """项目基础异常。"""

    stage: str = "unknown"
    retryable: bool = False

    def __init__(self, message: str, *, stage: str = "", retryable: bool = False, original: Exception | None = None):
        self.stage = stage or self.stage
        self.retryable = retryable or self.retryable
        self.original = original
        super().__init__(message)


class ConfigurationError(AgentProjectError):
    stage = "config"
    retryable = False


class ModelInvocationError(AgentProjectError):
    stage = "model"
    retryable = True


class RetrievalError(AgentProjectError):
    stage = "retrieval"
    retryable = True


class ToolExecutionError(AgentProjectError):
    stage = "tool"
    retryable = False


class MCPConnectionError(AgentProjectError):
    stage = "mcp"
    retryable = True


class DocumentParseError(AgentProjectError):
    stage = "document"
    retryable = False


class StreamExecutionError(AgentProjectError):
    stage = "stream"
    retryable = False


def to_safe_message(exc: AgentProjectError) -> dict:
    """将异常转换为安全的对外响应（不泄露内部信息）。"""
    return {
        "error": True,
        "stage": exc.stage,
        "message": str(exc),
        "retryable": exc.retryable,
    }
