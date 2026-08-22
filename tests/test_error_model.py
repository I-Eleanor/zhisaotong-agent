"""P1-4 统一错误模型测试。

核心验证：
- 统一错误响应只包含 error_code / safe_message / request_id 三个字段（全字段精确断言，
  防止重新加入旧版 error 兼容字段）；
- 项目异常由全局 handler 统一转换，使用异常自身的 error_code 与安全提示；
- 未知异常返回固定 500 / INTERNAL_ERROR / 通用安全提示，原始异常只进日志；
- request_id 与响应头 X-Request-ID 一致；
- /knowledge/rebuild、/knowledge/query、/chat/sync 的错误路径不泄漏内部信息
  （Chroma / 文件路径 / 密钥 / 用户输入原文）；
- 日志侧（P1-4.1 安全修正）：
  - 真实 logger 的控制台出口（生产 formatter）同样执行脱敏；
  - log_safe_text() 本身脱敏，且先脱敏后截断；
  - 全局异常 handler 不记录完整 traceback / 异常链原文，日志字段不含原始密钥。
"""
import logging

from fastapi.testclient import TestClient

from utils import error_codes
from utils.exceptions import (
    MCPConnectionError,
    ModelInvocationError,
    RetrievalError,
    StreamExecutionError,
    ToolExecutionError,
    normalize_error_code,
    safe_error_payload,
)


# ----------------------------------------------------------------- 安全载荷
def test_safe_error_payload_hides_exception_details():
    """普通异常的细节（含敏感信息）不得出现在客户端载荷中。"""
    exc = RuntimeError("DASHSCOPE_API_KEY=sk-secret-123 在连接时泄露")
    payload = safe_error_payload(exc, request_id="rid-abc")

    assert payload["error_code"] == error_codes.INTERNAL_ERROR
    assert payload["request_id"] == "rid-abc"
    assert payload["safe_message"], "应有面向用户的安全提示"
    serialized = str(payload)
    assert "sk-secret-123" not in serialized, "不得向客户端泄露异常细节"
    assert "RuntimeError" not in serialized


def test_safe_error_payload_maps_project_errors():
    """项目异常应映射到对应错误码与自身的安全提示，而不是内部错误。"""
    assert safe_error_payload(ModelInvocationError("x")) == {
        "error_code": error_codes.MODEL_UNAVAILABLE,
        "safe_message": "模型服务暂时不可用，请稍后重试",
        "request_id": "",
    }
    assert safe_error_payload(RetrievalError("x"))["error_code"] == error_codes.RETRIEVAL_FAILED
    assert safe_error_payload(ToolExecutionError("x"))["error_code"] == error_codes.TOOL_EXECUTION_FAILED
    assert safe_error_payload(MCPConnectionError("x"))["error_code"] == error_codes.TOOL_UNAVAILABLE
    assert safe_error_payload(StreamExecutionError("x"))["error_code"] == error_codes.STREAM_FAILED


def test_safe_error_payload_exact_fields_no_legacy_error():
    """统一错误载荷只有三个字段：不再保留旧版 error 兼容字段。"""
    payload = safe_error_payload(ValueError("内部细节"), request_id="r1")
    assert set(payload.keys()) == {"error_code", "safe_message", "request_id"}
    assert "error" not in payload, "不得重新加入原始 error 字段"
    assert "内部细节" not in str(payload)


def test_error_code_constants_are_stable():
    """错误码是稳定的字符串常量，客户端可据此分支处理。"""
    assert error_codes.MODEL_TIMEOUT == "MODEL_TIMEOUT"
    assert error_codes.MODEL_UNAVAILABLE == "MODEL_UNAVAILABLE"
    assert error_codes.RETRIEVAL_FAILED == "RETRIEVAL_FAILED"
    assert error_codes.TOOL_UNAVAILABLE == "TOOL_UNAVAILABLE"
    assert error_codes.INVALID_MODEL_OUTPUT == "INVALID_MODEL_OUTPUT"
    assert error_codes.STREAM_FAILED == "STREAM_FAILED"
    assert error_codes.CONTAINER_NOT_READY == "CONTAINER_NOT_READY"


def test_container_state_error_payload_is_safe():
    """ContainerStateError：稳定错误码 + 固定安全提示，内部状态与异常原文不出现在载荷。"""
    from utils.exceptions import ContainerStateError

    exc = ContainerStateError("容器正在关闭中（CLOSING），资源=chat_model，线程=Tid-123")
    payload = safe_error_payload(exc, request_id="rid-1")

    assert payload["error_code"] == error_codes.CONTAINER_NOT_READY
    assert payload["safe_message"] == "服务尚未就绪，请稍后重试"
    serialized = str(payload)
    for forbidden in ("CLOSING", "chat_model", "Tid-123", "容器正在关闭"):
        assert forbidden not in serialized, f"安全载荷不得携带内部信息: {forbidden}"


def test_normalize_error_code_rejects_unknown_values():
    """事件等外部来源的 error_code 必须过滤：未知值回退 INTERNAL_ERROR。"""
    assert normalize_error_code("MODEL_UNAVAILABLE") == error_codes.MODEL_UNAVAILABLE
    assert normalize_error_code("boom: sk-secret-123") == error_codes.INTERNAL_ERROR
    assert normalize_error_code("") == error_codes.INTERNAL_ERROR
    assert normalize_error_code(None) == error_codes.INTERNAL_ERROR
    assert normalize_error_code(123) == error_codes.INTERNAL_ERROR


# ----------------------------------------------------------------- /chat/sync 错误路径
def test_chat_sync_error_event_returns_structured_payload(api_client):
    """Agent 输出 error 事件时，/chat/sync 返回统一错误载荷（全字段精确断言）。"""

    class ErrorOrchestrator:
        def execute(self, query, history=None, mode=None):
            yield {
                "type": "error",
                "agent": "conversation",
                "content": "对话处理失败，请稍后重试。",
                "data": {"error_code": error_codes.MODEL_UNAVAILABLE, "request_id": "r-evt"},
            }
            yield {"type": "done", "agent": "conversation", "content": ""}

    from api.main import app

    app.state.container._orchestrator = ErrorOrchestrator()
    resp = api_client.post(
        "/api/chat/sync", json={"query": "触发错误"}, headers={"X-Request-ID": "rid-evt-1"}
    )

    assert resp.status_code == 500
    assert resp.json() == {
        "error_code": error_codes.MODEL_UNAVAILABLE,
        "safe_message": "对话处理失败，请稍后重试。",
        "request_id": "rid-evt-1",
    }
    assert resp.headers["X-Request-ID"] == "rid-evt-1", "request_id 应与响应头一致"


def test_chat_sync_error_event_does_not_leak_sensitive_content(api_client):
    """error 事件内容不可信：content 含密钥 / 用户原文、error_code 含异常文本时不得透传。"""
    secret = "sk-POISON-1234567890"

    class PoisonedOrchestrator:
        def execute(self, query, history=None, mode=None):
            yield {
                "type": "error",
                "agent": "conversation",
                "content": f"内部错误：api_key={secret} 回显[{query}]",
                "data": {"error_code": f"boom: {secret}", "request_id": "r"},
            }
            yield {"type": "done", "agent": "conversation", "content": ""}

    from api.main import app

    app.state.container._orchestrator = PoisonedOrchestrator()
    resp = api_client.post(
        "/api/chat/sync", json={"query": "用户查询原文"}, headers={"X-Request-ID": "rid-poison-1"}
    )

    assert resp.status_code == 500
    body = resp.json()
    assert body == {
        "error_code": error_codes.INTERNAL_ERROR,  # 未知码回退
        "safe_message": "对话处理失败，请稍后重试。",
        "request_id": "rid-poison-1",
    }
    assert secret not in str(body), "不得泄漏事件 content / error_code 中的敏感文本"
    assert "用户查询原文" not in str(body), "客户端传来的内容不得进入安全提示"


def test_chat_sync_exception_returns_safe_payload(api_client):
    """Orchestrator 抛出异常时，客户端只收到统一安全载荷，异常细节仅进日志。"""
    secret = "sk-leaked-key-456"

    class BrokenOrchestrator:
        def execute(self, query, history=None, mode=None):
            raise RuntimeError(f"连接失败，密钥 {secret}")
            yield  # pragma: no cover

    from api.main import app

    app.state.container._orchestrator = BrokenOrchestrator()
    resp = api_client.post(
        "/api/chat/sync", json={"query": "触发异常"}, headers={"X-Request-ID": "rid-exc-1"}
    )

    assert resp.status_code == 500
    assert resp.json() == {
        "error_code": error_codes.INTERNAL_ERROR,
        "safe_message": "服务暂时异常，请稍后重试",
        "request_id": "rid-exc-1",
    }
    assert secret not in str(resp.json()), "异常细节不得泄露给客户端"
    assert resp.headers["X-Request-ID"] == "rid-exc-1"


def test_chat_sync_project_exception_keeps_own_code(api_client):
    """Agent 抛出的项目异常保留自身错误码，不被包装成 INTERNAL_ERROR。"""

    class ModelBrokenOrchestrator:
        def execute(self, query, history=None, mode=None):
            raise ModelInvocationError(f"模型连接失败 sk-inner-777: {query}")
            yield  # pragma: no cover

    from api.main import app

    app.state.container._orchestrator = ModelBrokenOrchestrator()
    resp = api_client.post("/api/chat/sync", json={"query": "模型挂了"})

    assert resp.status_code == 500
    assert resp.json() == {
        "error_code": error_codes.MODEL_UNAVAILABLE,
        "safe_message": "模型服务暂时不可用，请稍后重试",
        "request_id": resp.json()["request_id"],
    }
    assert "sk-inner-777" not in str(resp.json())


# ----------------------------------------------------------------- 知识库接口错误路径
def test_knowledge_query_project_error_maps_code(api_client):
    """RAG 抛项目异常（RetrievalError）→ 全局 handler 用异常自身错误码与提示。"""

    class RagRaisesProject:
        def rag_with_sources(self, query):
            raise RetrievalError(f"chroma 内部错误 sk-INNER-999: {query}")

    from api.main import app

    app.state.container._rag_service = RagRaisesProject()
    resp = api_client.post(
        "/api/knowledge/query", json={"query": "扫地机"}, headers={"X-Request-ID": "rid-proj-1"}
    )

    assert resp.status_code == 500
    assert resp.json() == {
        "error_code": error_codes.RETRIEVAL_FAILED,
        "safe_message": "知识检索暂时不可用，请稍后重试",
        "request_id": "rid-proj-1",
    }
    assert "sk-INNER-999" not in str(resp.json())


def test_knowledge_query_internal_error_no_leak(api_client):
    """RAG 内部异常（Chroma / 模型文本）不得出现在响应中。"""

    class BrokenRag:
        def rag_with_sources(self, query):
            raise RuntimeError("chroma 连接失败 D:\\secret\\db 路径 sk-999")

    from api.main import app

    app.state.container._rag_service = BrokenRag()
    resp = api_client.post(
        "/api/knowledge/query", json={"query": "扫地机"}, headers={"X-Request-ID": "rid-q-1"}
    )

    assert resp.status_code == 500
    body = resp.json()
    assert body == {
        "error_code": error_codes.INTERNAL_ERROR,
        "safe_message": "服务暂时异常，请稍后重试",
        "request_id": "rid-q-1",
    }
    serialized = str(body)
    assert "chroma" not in serialized.lower(), "不得泄漏原始 Chroma 异常文本"
    assert "D:\\secret" not in serialized, "不得泄漏文件系统路径"
    assert "sk-999" not in serialized


def test_knowledge_rebuild_internal_error_no_leak(api_client):
    """重建向量库的内部异常（文件系统 / Chroma）不得出现在响应中。"""

    class BrokenVS:
        def load_document(self):
            raise RuntimeError("chroma 读取失败：D:\\data\\knowledge 路径，api_key=sk-777")

        def count(self):
            return 1

    from api.main import app

    app.state.container._vector_store = BrokenVS()
    resp = api_client.post(
        "/api/knowledge/rebuild", headers={"X-Request-ID": "rid-reb-1"}
    )

    assert resp.status_code == 500
    body = resp.json()
    assert body == {
        "error_code": error_codes.INTERNAL_ERROR,
        "safe_message": "服务暂时异常，请稍后重试",
        "request_id": "rid-reb-1",
    }
    serialized = str(body)
    assert "chroma" not in serialized.lower()
    assert "D:\\data" not in serialized
    assert "sk-777" not in serialized


# ----------------------------------------------------------------- 全局未知异常处理
def test_unhandled_exception_returns_fixed_500():
    """未包装的未知异常：固定 500 / INTERNAL_ERROR / 通用提示，request_id 与响应头一致。"""
    from api.main import app

    @app.get("/api/_test/unhandled")
    def _unhandled():
        raise RuntimeError("内部异常：api_key=sk-ROOT-555 泄漏")

    # 未知异常经 ServerErrorMiddleware 处理后会向上重抛，测试客户端需关闭重抛才能收到响应
    client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = client.get("/api/_test/unhandled", headers={"X-Request-ID": "rid-unhandled-1"})
        assert resp.status_code == 500
        assert resp.json() == {
            "error_code": error_codes.INTERNAL_ERROR,
            "safe_message": "服务暂时异常，请稍后重试",
            "request_id": "rid-unhandled-1",
        }
        assert resp.headers["X-Request-ID"] == "rid-unhandled-1", \
            "未知异常路径下用户层中间件被跳过，handler 需自行回写 X-Request-ID"
        assert "sk-ROOT-555" not in str(resp.json())
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes
            if getattr(r, "path", "") != "/api/_test/unhandled"
        ]


# ----------------------------------------------------------------- 错误日志
def test_error_handler_log_contains_request_id_type_stage(api_client, caplog):
    """全局 handler 的日志应包含 request_id / error_type / stage，便于排查。"""

    class BrokenOrchestrator:
        def execute(self, query, history=None, mode=None):
            raise RuntimeError(f"连接失败 sk-log-key-321: {query}")
            yield  # pragma: no cover

    from api.main import app

    app.state.container._orchestrator = BrokenOrchestrator()
    with caplog.at_level(logging.WARNING, logger="agent"):
        resp = api_client.post(
            "/api/chat/sync", json={"query": "q"}, headers={"X-Request-ID": "rid-log-1"}
        )

    assert resp.status_code == 500
    entries = [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "unhandled_project_error"
    ]
    assert entries, "全局 handler 应记录结构化日志"
    entry = entries[-1]
    assert entry["request_id"] == "rid-log-1"
    assert entry["error_type"] == "AgentProjectError"
    assert entry["original_error_type"] == "RuntimeError"
    assert entry["stage"] == "sync_chat"


def test_json_formatter_redacts_secrets_in_log_output():
    """日志清洗回归：文件日志格式化时对密钥类敏感文本脱敏。"""
    from utils.logger_handler import JsonFormatter

    record = logging.LogRecord(
        name="agent",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg={"event": "unhandled_exception", "error_msg": "api_key=sk-REDACT-123456 连接失败"},
        args=(),
        exc_info=None,
        func="test",
    )
    out = JsonFormatter().format(record)

    assert "sk-REDACT-123456" not in out, "敏感密钥不得原样写入日志"
    assert "***REDACTED***" in out


# ----------------------------------------------------------------- 日志侧脱敏回归（P1-4.1）
def test_log_safe_text_redacts_secrets():
    """log_safe_text 是调用侧第一道防线：密钥类文本必须脱敏。"""
    from utils.logger_handler import log_safe_text

    out = log_safe_text("连接失败 api_key=sk-SAFETEXT-1234567890 token=TOK-SAFETEXT-9988")
    assert "sk-SAFETEXT-1234567890" not in out
    assert "TOK-SAFETEXT-9988" not in out
    assert out.count("***REDACTED***") == 2


def test_log_safe_text_redacts_before_truncation():
    """先脱敏后截断：截断边界不会把密钥切成正则匹配不到的残段。"""
    from utils.logger_handler import log_safe_text

    filler = "填充" * 60  # 120 字符，超过默认截断长度 100
    out = log_safe_text(f"{filler} api_key=sk-EDGE-998877665544")
    assert "sk-EDGE-998877665544" not in out, "截断发生在脱敏之后，密钥不得以残段形式存活"


def test_real_logger_console_output_redacts_secrets():
    """回归：真实 agent logger 的控制台出口（生产 formatter）执行脱敏，不只文件出口。"""
    import io

    from utils import logger_handler
    from utils.logger_handler import logger

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logger_handler.CONSOLE_FORMAT)
    logger.addHandler(handler)
    try:
        logger.error({
            "event": "unhandled_exception",
            "stage": "api",
            "error_msg": "连接失败 api_key=sk-CONSOLE-123456789 token=CONSOLE-TOKEN-998877 secret=CONSOLE-SECRET-445566",
        })
        logger.error("纯文本消息 secret=PLAINSECRET-77889900")
    finally:
        logger.removeHandler(handler)

    out = stream.getvalue()
    assert "api_key=***REDACTED***" in out, "控制台输出必须包含脱敏标记"
    assert "token=***REDACTED***" in out
    assert "secret=***REDACTED***" in out
    assert "sk-CONSOLE-123456789" not in out
    assert "CONSOLE-TOKEN-998877" not in out
    assert "CONSOLE-SECRET-445566" not in out
    assert "PLAINSECRET-77889900" not in out


def test_global_handler_log_redacts_and_omits_traceback(api_client, caplog):
    """全局项目异常日志：字段不含原始密钥、无 traceback 字段、无异常链原文。"""
    secret = "sk-HANDLER-112233445566"

    class BrokenOrchestrator:
        def execute(self, query, history=None, mode=None):
            raise RuntimeError(f"api_key={secret} 连接失败")
            yield  # pragma: no cover

    from api.main import app

    app.state.container._orchestrator = BrokenOrchestrator()
    with caplog.at_level(logging.WARNING, logger="agent"):
        resp = api_client.post(
            "/api/chat/sync", json={"query": "q"}, headers={"X-Request-ID": "rid-51-a"}
        )

    assert resp.status_code == 500
    entries = [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "unhandled_project_error"
    ]
    assert entries, "全局项目异常 handler 应记录结构化日志"
    entry = entries[-1]
    serialized = str(entry)
    assert secret not in serialized, "日志任何字段不得含原始密钥"
    assert "api_key=***REDACTED***" in entry["original_error_msg"]
    assert "traceback" not in entry, "不得记录完整 traceback 字段"
    assert "Traceback (most recent call last)" not in serialized
    assert "During handling of the above exception" not in serialized, "不得记录异常链原文"
    assert "The above exception was the direct cause" not in serialized


def test_unhandled_exception_log_is_summary_without_traceback(caplog):
    """全局未知异常日志：只记录清洗后的摘要字段，无完整 traceback / 异常链。"""
    from api.main import app

    @app.get("/api/_test/unhandled_log")
    def _unhandled_log():
        raise RuntimeError("boom api_key=sk-RAWLOG-887766554433")

    client = TestClient(app, raise_server_exceptions=False)
    try:
        with caplog.at_level(logging.WARNING, logger="agent"):
            resp = client.get("/api/_test/unhandled_log", headers={"X-Request-ID": "rid-51-b"})

        assert resp.status_code == 500
        entries = [
            r.msg for r in caplog.records
            if isinstance(r.msg, dict) and r.msg.get("event") == "unhandled_exception"
        ]
        assert entries, "全局未知异常 handler 应记录结构化日志"
        entry = entries[-1]
        serialized = str(entry)
        assert "sk-RAWLOG-887766554433" not in serialized
        assert "api_key=***REDACTED***" in entry["error_msg"], "error_msg 应为脱敏后的摘要"
        assert "traceback" not in entry, "不得记录完整 traceback 字段"
        assert "Traceback (most recent call last)" not in serialized
        assert "During handling of the above exception" not in serialized
        assert entry["error_type"] == "RuntimeError"
        assert entry["stage"] == "api"
        assert entry["request_id"] == "rid-51-b"
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes
            if getattr(r, "path", "") != "/api/_test/unhandled_log"
        ]


# ----------------------------------------------------------------- 模型工厂安全（P1-12）
def test_chat_model_factory_init_failure_hides_key(monkeypatch, caplog):
    """模型构造失败：异常原文（含密钥/路径）只进脱敏日志，客户端只见 ModelInvocationError。"""
    import logging

    from model.factory import _build_chat_model
    from utils.exceptions import ModelInvocationError
    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter

    secret = "sk-FACTORY-998877665544"
    abs_path = "D:\\secret\\factory\\chat.bin"

    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    monkeypatch.setattr(
        "model.factory.ChatOpenAI",
        lambda **kw: (_ for _ in ()).throw(RuntimeError(f"初始化失败 api_key={secret} {abs_path}")),
    )

    with caplog.at_level(logging.ERROR, logger="agent"):
        try:
            _build_chat_model()
            raise AssertionError("应抛出 ModelInvocationError")
        except ModelInvocationError as exc:
            assert secret not in str(exc), "异常消息不得含密钥原文"
            assert abs_path not in str(exc)
            assert exc.original is not None, "原始异常应保留在 original 字段供日志排查"

    entries = [r.msg for r in caplog.records
               if isinstance(r.msg, dict) and r.msg.get("event") == "chat_model_init_error"]
    assert entries, "应记录模型初始化失败事件"
    json_outs = [JsonFormatter().format(r) for r in caplog.records]
    console_outs = [CONSOLE_FORMAT.format(r) for r in caplog.records]
    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert secret not in combined, f"{label}：初始化日志不得泄漏密钥"
        assert abs_path not in combined, f"{label}：初始化日志不得泄漏路径"


def test_chat_model_factory_missing_key_error_is_safe(monkeypatch):
    """密钥未设置：报错消息只含变量名提示，不含任何密钥形态。"""
    from model.factory import _build_chat_model

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    try:
        _build_chat_model()
        raise AssertionError("应抛出异常")
    except Exception as exc:
        msg = str(exc)
        assert "DEEPSEEK_API_KEY" in msg
        assert "sk-" not in msg, "未设置密钥的报错不得出现密钥形态"


def test_fake_key_placeholder_rejected(monkeypatch):
    """示例占位密钥（your_deepseek_api_key_here）被校验层拒绝，不得进入模型构造。"""
    from utils.config_validator import ConfigValidationError, validate_before_use

    monkeypatch.setenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here")
    try:
        validate_before_use("chat_model")
        raise AssertionError("占位密钥应被拒绝")
    except ConfigValidationError as exc:
        assert "your_deepseek_api_key_here" not in str(exc), "报错不得回显占位密钥值"
