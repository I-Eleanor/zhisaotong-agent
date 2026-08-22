"""统一业务异常体系。

所有自定义异常继承自 AgentProjectError，按阶段分类：
- ConfigurationError：配置缺失或无效
- ModelInvocationError：LLM / Embedding 调用失败
- RetrievalError：向量检索失败
- ToolExecutionError：工具执行失败
- ServiceUnavailableError：底层服务 / 数据不可用
- MCPConnectionError：MCP Server 连接失败
- DocumentParseError：文档解析失败
- StreamExecutionError：SSE 流式输出异常
- ContainerStateError：容器状态不允许当前操作（CLOSING / CLOSED / 未挂载）

每个异常携带：
- stage：出错阶段（config / model / retrieval / tool / service / mcp / document / stream）
- retryable：是否可重试
- error_code：稳定的客户端可见错误码（utils.error_codes）
- original：原始异常（不暴露给前端）
"""
from utils import error_codes

# 通用兜底安全提示：不携带任何内部细节
_GENERIC_SAFE_MESSAGE = "服务暂时异常，请稍后重试"

# 已知稳定错误码集合：error_code 只允许输出其中的值，未知来源一律回退 INTERNAL_ERROR
KNOWN_ERROR_CODES = frozenset(
    value
    for key, value in vars(error_codes).items()
    if isinstance(value, str) and key.isupper()
)


class AgentProjectError(Exception):
    """项目基础异常。

    message 只进服务端日志；safe_message 是可展示给用户的安全提示（默认通用文案）。
    """

    stage: str = "unknown"
    retryable: bool = False
    error_code: str = error_codes.INTERNAL_ERROR
    safe_message: str = _GENERIC_SAFE_MESSAGE

    def __init__(self, message: str, *, stage: str = "", retryable: bool = False, original: Exception | None = None):
        self.stage = stage or self.stage
        self.retryable = retryable or self.retryable
        self.original = original
        super().__init__(message)


class ConfigurationError(AgentProjectError):
    stage = "config"
    retryable = False
    error_code = error_codes.INTERNAL_ERROR


class ModelInvocationError(AgentProjectError):
    stage = "model"
    retryable = True
    error_code = error_codes.MODEL_UNAVAILABLE
    safe_message = "模型服务暂时不可用，请稍后重试"


class RetrievalError(AgentProjectError):
    stage = "retrieval"
    retryable = True
    error_code = error_codes.RETRIEVAL_FAILED
    safe_message = "知识检索暂时不可用，请稍后重试"


class ToolExecutionError(AgentProjectError):
    stage = "tool"
    retryable = False
    error_code = error_codes.TOOL_EXECUTION_FAILED
    safe_message = "工具执行失败，请稍后重试"


class ServiceUnavailableError(AgentProjectError):
    """底层服务 / 数据不可用。

    底层服务在无法提供有效数据时抛出（如设备数据未查询到），
    由 ToolRouter 捕获并转为 StepResult(success=False)——
    错误字符串返回值会被上层误当成成功结果。
    """

    stage = "service"
    retryable = True
    error_code = error_codes.SERVICE_UNAVAILABLE
    safe_message = "服务暂时不可用，请稍后重试"


class MCPConnectionError(AgentProjectError):
    stage = "mcp"
    retryable = True
    error_code = error_codes.TOOL_UNAVAILABLE
    safe_message = "外部服务连接失败，请稍后重试"


class DocumentParseError(AgentProjectError):
    stage = "document"
    retryable = False
    error_code = error_codes.INTERNAL_ERROR
    safe_message = "文档解析失败，请稍后重试"


class StreamExecutionError(AgentProjectError):
    stage = "stream"
    retryable = False
    error_code = error_codes.STREAM_FAILED
    safe_message = "响应流中断，请稍后重试"


class ContainerStateError(AgentProjectError):
    """容器状态不允许当前操作（CLOSING 期间访问资源、容器未挂载等）。

    message 只进服务端日志；safe_message 为固定通用文案，不携带容器
    状态、资源名、线程信息或异常原文。
    """

    stage = "service"
    retryable = True
    error_code = error_codes.CONTAINER_NOT_READY
    safe_message = "服务尚未就绪，请稍后重试"


def safe_error_payload(exc: Exception, request_id: str = "", default_code: str = error_codes.INTERNAL_ERROR) -> dict:
    """把任意异常转换为对客户端安全的结构化载荷。

    统一错误响应只包含 error_code / safe_message / request_id 三个字段；
    项目异常使用自身的错误码与安全提示，其他异常回退默认码与通用提示。
    原始异常信息由调用方负责写入日志。
    """
    if isinstance(exc, AgentProjectError):
        code = exc.error_code
        message = exc.safe_message
    else:
        code = default_code
        message = _GENERIC_SAFE_MESSAGE
    return {
        "error_code": code,
        "safe_message": message,
        "request_id": request_id,
    }


def normalize_error_code(value: object, default: str = error_codes.INTERNAL_ERROR) -> str:
    """把任意来源的 error_code 归一化为已知稳定错误码，未知值回退默认码。

    Agent 事件等外部数据里的 error_code 无法保证可信（可能混入异常原文），
    输出前必须经此函数过滤。
    """
    if isinstance(value, str) and value in KNOWN_ERROR_CODES:
        return value
    return default
