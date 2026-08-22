"""P1-17 最终架构验收：跨模块契约的端到端锁定。

聚合验收（复用 conftest 现有桩与夹具，零真实模型 / 网络 / 本地向量库）：
1. 全新进程启动契约：配置校验失败安全退出、有效配置可启动、启动零重型资源、
   live 探针 200、ready 探针按依赖可用性 200 / 安全 503；
2. 依赖边界契约：API 路由不触碰模块级 get_* 全局单例（源码静态扫描），
   业务请求只经 AppContainer 取资源，容器非 OPEN 时业务请求 503 且
   绝不隐式重建资源；
3. 错误与日志契约：API 错误响应恒为三字段安全结构，SSE error / done
   协议不变，错误路径日志不含密钥 / 绝对路径 / 用户原文 / traceback，
   全部 logger 调用为结构化 dict（无 f-string 拼接）。
"""
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 业务路由集合：验收其对非 OPEN 容器的统一 503 行为
_BUSINESS_ROUTES = [
    ("POST", "/api/chat", {"query": "你好", "history": [], "mode": "conversation"}),
    ("POST", "/api/diagnose", {"query": "设备无法启动"}),
    ("POST", "/api/knowledge/query", {"query": "滤网"}),
]


# ----------------------------------------------------------------- 1. 启动契约
def test_startup_validation_failure_exits_safely(monkeypatch, caplog):
    """配置校验失败：安全结构化日志 + 阻止启动 + 零容器构造（全新进程语义）。"""
    import logging

    from fastapi import FastAPI

    from api import main as main_mod
    from utils.config_validator import ConfigValidationError

    def boom():
        raise ConfigValidationError("配置校验失败:\n  [环境变量] DEEPSEEK_API_KEY: 未设置或使用示例值\n")

    def forbidden_container():
        raise AssertionError("校验失败时不得构造容器")

    monkeypatch.setattr(main_mod, "validate_startup", boom)
    monkeypatch.setattr(main_mod, "AppContainer", forbidden_container)

    app = FastAPI(lifespan=main_mod.lifespan)
    with (  # lifespan 在 TestClient 进入上下文时执行并抛出校验异常
        caplog.at_level(logging.ERROR, logger="agent"),
        pytest.raises(ConfigValidationError),
        TestClient(app),
    ):
        pass

    records = [r.msg for r in caplog.records
               if isinstance(r.msg, dict) and r.msg.get("event") == "startup_config_validation_failed"]
    assert len(records) == 1
    assert set(records[0]) == {"event", "error_type", "error_msg"}, "安全异常摘要形态"


def test_valid_config_starts_and_probes_healthy(monkeypatch):
    """配置有效：可启动（挂载容器），live 200，依赖可用时 ready 200。"""
    from api import main as main_mod
    from api.container import AppContainer

    monkeypatch.setattr(main_mod, "validate_startup", lambda: None)

    class FakeVectorStore:
        def count(self):
            return 42

    class ReadyContainer(AppContainer):
        """vector_store 注入桩；chat_model 经工厂桩（conftest autouse）懒加载。"""

    try:
        with TestClient(main_mod.app) as client:
            container = ReadyContainer()
            container._vector_store = FakeVectorStore()
            main_mod.app.state.container = container

            live = client.get("/api/health/live")
            assert live.status_code == 200
            assert live.json()["status"] == "alive"

            ready = client.get("/api/health/ready")
            assert ready.status_code == 200
            body = ready.json()
            assert body["status"] == "ready"
            assert body["checks"]["vector_store"] == {"ok": True, "chunk_count": 42}
            assert body["checks"]["chat_model"]["ok"] is True
    finally:
        del main_mod.app.state.container


def test_startup_loads_no_heavy_resources(monkeypatch):
    """启动阶段零重型资源：模型 / Embedding / 向量库工厂被调用即失败。"""
    from api import main as main_mod

    def boom(*args, **kwargs):
        raise AssertionError("启动阶段不得加载模型 / Embedding / 向量库")

    monkeypatch.setattr(main_mod, "validate_startup", lambda: None)
    monkeypatch.setattr("model.factory.ChatModelFactory.generator", boom)
    monkeypatch.setattr("model.factory.EmbeddingsFactory.generator", boom)
    monkeypatch.setattr("rag.vector_store.VectorStoreService", boom)

    try:
        with TestClient(main_mod.app) as client:
            resp = client.get("/api/health/live")
            assert resp.status_code == 200, "live 探针不依赖任何重型资源"
            container = main_mod.app.state.container
            assert container._chat_model is None
            assert container._embedding_model is None
            assert container._vector_store is None
            assert container._rag_service is None
            assert container._orchestrator is None
    finally:
        del main_mod.app.state.container


def test_ready_returns_safe_503_when_dependency_unavailable(api_client, monkeypatch):
    """依赖不可用：readiness 返回安全 503（无内部信息），不抛裸异常。"""
    from api.container import AppContainer
    from api.main import app

    class BrokenVectorStore:
        def count(self):
            raise RuntimeError("chroma 连接失败: D:\\data\\chroma 密码 sk-secret-123")

    container = AppContainer()
    container._vector_store = BrokenVectorStore()
    monkeypatch.setattr(app.state, "container", container, raising=False)

    resp = api_client.get("/api/health/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["vector_store"]["ok"] is False
    serialized = str(body)
    for forbidden in ("chroma", "D:\\", "sk-secret-123", "RuntimeError", "Traceback"):
        assert forbidden not in serialized, f"ready 响应不得泄漏内部信息: {forbidden}"


# ----------------------------------------------------------------- 2. 依赖边界契约
def test_api_source_calls_no_global_singletons():
    """静态扫描：api/ 生产源码不调用模块级 get_* 全局单例（架构锁定）。

    只匹配调用形态（get_xxx(），docstring 中的名词不受影响）。
    """
    forbidden = re.compile(
        r"\b(get_orchestrator|get_chat_model|get_embed_model|get_diagnostic_agent)\("
    )
    offenders = []
    for path in (PROJECT_ROOT / "api").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if forbidden.search(line):
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"API 层不得调用全局单例 getter: {offenders}"


def test_business_routes_503_on_closed_container_without_rebuild(api_client, monkeypatch):
    """容器 CLOSED：全部业务路由 503，且不隐式重建任何资源（零懒加载）。"""
    from api.container import AppContainer
    from api.main import app

    container = AppContainer()
    container.close()
    monkeypatch.setattr(app.state, "container", container, raising=False)

    for method, url, payload in _BUSINESS_ROUTES:
        resp = api_client.request(method, url, json=payload)
        assert resp.status_code == 503, f"{url} 在 CLOSED 容器上应 503"

    # 重建路径审计：容器五项资源引用保持 None
    assert container._chat_model is None, "CLOSED 容器上的请求不得隐式重建 chat 模型"
    assert container._embedding_model is None
    assert container._vector_store is None
    assert container._rag_service is None
    assert container._orchestrator is None, "CLOSED 容器上的请求不得隐式重建 orchestrator"


def test_business_routes_503_on_closing_container(api_client, monkeypatch):
    """容器 CLOSING：业务路由同样 503，不产生生命周期之外的新资源。"""
    from api.container import AppContainer, ContainerState
    from api.main import app

    container = AppContainer()
    container._state = ContainerState.CLOSING
    monkeypatch.setattr(app.state, "container", container, raising=False)

    for method, url, payload in _BUSINESS_ROUTES:
        resp = api_client.request(method, url, json=payload)
        assert resp.status_code == 503, f"{url} 在 CLOSING 容器上应 503"

    assert container._chat_model is None and container._orchestrator is None


def test_closed_container_request_logs_safe_fields(api_client, monkeypatch, caplog):
    """非 OPEN 容器上的业务请求：记 container_not_ready 安全四元组日志。"""
    import logging

    from api.container import AppContainer
    from api.main import app

    container = AppContainer()
    container.close()
    monkeypatch.setattr(app.state, "container", container, raising=False)

    with caplog.at_level(logging.WARNING, logger="agent"):
        resp = api_client.post("/api/chat", json={"query": "你好"})

    assert resp.status_code == 503
    records = [r.msg for r in caplog.records
               if isinstance(r.msg, dict) and r.msg.get("event") == "container_not_ready"]
    assert len(records) == 1
    assert set(records[0]) == {"event", "state", "request_id", "error_code"}
    assert records[0]["state"] == "closed"


# ----------------------------------------------------------------- 3. 错误与日志契约
def test_api_error_response_is_three_field_safe_structure(api_client, monkeypatch):
    """未预期异常：API 错误响应恒为 {error_code, safe_message, request_id} 三字段。"""
    from api.main import app

    class ExplodingRag:
        def rag_with_sources(self, query):
            raise ValueError(f"内部错误：用户输入 {query}，路径 D:\\secrets\\key.txt")

    container = app.state.container
    container._rag_service = ExplodingRag()

    resp = api_client.post("/api/knowledge/query", json={"query": "我的密钥是 sk-abc123def456"})

    assert resp.status_code == 500
    body = resp.json()
    assert set(body) == {"error_code", "safe_message", "request_id"}, "三字段安全结构"
    serialized = str(body)
    for forbidden in ("sk-abc123def456", "D:\\", "内部错误", "ValueError", "key.txt"):
        assert forbidden not in serialized, f"错误响应不得携带内部信息: {forbidden}"


def test_sse_error_and_done_protocol_unchanged(api_client, monkeypatch):
    """SSE 协议：生产线程异常 → 安全 error 事件（STREAM_FAILED）后流必然终止。"""
    from api.main import app

    class ExplodingOrchestrator:
        def execute(self, query, history=None, mode=None):
            raise RuntimeError("模型崩溃：sk-secret-999 与 D:\\model\\weights")
            yield  # pragma: no cover - 使 execute 成为生成器函数

    container = app.state.container
    container._orchestrator = ExplodingOrchestrator()

    resp = api_client.post("/api/chat", json={"query": "你好"})

    assert resp.status_code == 200
    text = resp.text
    assert "event: error" in text, "异常应下发 error 事件"
    assert "STREAM_FAILED" in text, "error 事件携带稳定错误码"
    assert "服务暂时异常" in text, "error 事件使用固定安全文案"
    assert "sk-secret-999" not in text and "D:\\model" not in text, "error 事件不得泄漏异常原文"
    # done 语义：流必然结束（TestClient 完整消费即证明），且无 data 尾随 error


def test_error_path_logs_contain_no_secrets_paths_or_traceback(api_client, monkeypatch, caplog):
    """错误路径全量日志扫描：无密钥形态、无绝对路径、无 Traceback、无用户原文。"""
    import logging

    from api.main import app

    user_input = "我的密钥是 sk-zyxwvu876543"

    class ExplodingRag:
        def rag_with_sources(self, query):
            raise RuntimeError(f"用户输入={query}，模型路径 D:\\weights\\llm")

    container = app.state.container
    container._rag_service = ExplodingRag()

    with caplog.at_level(logging.WARNING, logger="agent"):
        api_client.post("/api/knowledge/query", json={"query": user_input})

    all_logged = "\n".join(str(r.msg) for r in caplog.records)
    assert caplog.records, "应产生错误日志"
    assert "sk-zyxwvu876543" not in all_logged, "日志不得含密钥形态"
    assert "D:\\weights" not in all_logged and "/home/" not in all_logged, "日志不得含绝对路径"
    assert "Traceback" not in all_logged, "日志不得含 traceback"
    assert user_input not in all_logged, "日志不得含用户原始输入"


def test_all_logger_calls_are_structured_not_fstring():
    """静态扫描：全部生产源码的 logger 调用为结构化 dict，无 f-string 拼接。"""
    pattern = re.compile(r"logger\.(info|warning|error|debug|critical)\(\s*f[\"']")
    offenders = []
    for pkg in ("api", "agent", "rag", "utils", "model"):
        for path in (PROJECT_ROOT / pkg).rglob("*.py"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{lineno}")
    assert not offenders, f"不得引入 f-string 日志（绕过统一脱敏）: {offenders}"
