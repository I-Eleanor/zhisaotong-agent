"""API 接口测试：health / chat(SSE) / diagnose(SSE) / knowledge upload / knowledge rebuild。

对话与诊断接口在路由层把 get_orchestrator 替换为固定事件流（见 conftest.api_client），
避免构造需要真实模型的 ConversationAgent。上传/重建接口额外打桩以隔离文件系统与向量库。
"""
import contextlib
import json
import os

import pytest


def test_health(api_client):
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "reranker_enabled" in body


def test_health_live(api_client):
    """存活探针：进程存活即 200，不做依赖检查。"""
    resp = api_client.get("/api/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def _make_ready_container(vector_store=None):
    """构造注入了测试替身的容器，供就绪探针测试使用。"""
    from api.container import AppContainer

    container = AppContainer()
    if vector_store is not None:
        container._vector_store = vector_store
    return container


def test_health_ready_success(api_client, monkeypatch):
    """就绪探针：所有必要依赖可用时返回 200 和各项检查结果。"""
    from api.main import app

    class FakeVectorStore:
        def count(self):
            return 42

    monkeypatch.setattr(app.state, "container", _make_ready_container(FakeVectorStore()), raising=False)
    resp = api_client.get("/api/health/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["checks"]["vector_store"] == {"ok": True, "chunk_count": 42}
    assert body["checks"]["chat_model"]["ok"] is True


def test_health_ready_failure_returns_503(api_client, monkeypatch):
    """就绪探针：任一必要依赖不可用时返回 503。"""
    from api.main import app

    class BrokenVectorStore:
        def count(self):
            raise RuntimeError("chroma 不可用")

    monkeypatch.setattr(app.state, "container", _make_ready_container(BrokenVectorStore()), raising=False)
    resp = api_client.get("/api/health/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["vector_store"]["ok"] is False
    assert body["checks"]["chat_model"]["ok"] is True


def test_health_ready_slow_dependency_times_out(api_client, monkeypatch):
    """就绪探针：慢依赖在短超时内快速失败，返回 503 而非长时间阻塞。"""
    import time as time_mod

    from api import main as main_mod
    from api.main import app

    class SlowVectorStore:
        def count(self):
            time_mod.sleep(0.5)
            return 1

    monkeypatch.setattr(main_mod, "READY_CHECK_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(app.state, "container", _make_ready_container(SlowVectorStore()), raising=False)

    resp = api_client.get("/api/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["checks"]["vector_store"]["ok"] is False
    assert "超时" in body["checks"]["vector_store"]["error"]


def test_chat_sse_stream(api_client):
    resp = api_client.post("/api/chat", json={"query": "你好", "history": [], "mode": "conversation"})
    assert resp.status_code == 200
    text = resp.text
    assert "event: message" in text
    assert '"done"' in text


def test_diagnose_sse_stream(api_client):
    resp = api_client.post("/api/diagnose", json={"query": "设备无法启动"})
    assert resp.status_code == 200
    assert "event: message" in resp.text
    assert '"done"' in resp.text


def test_knowledge_upload(api_client, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routes.knowledge.get_abs_path", lambda p: str(tmp_path))
    resp = api_client.post(
        "/api/knowledge/upload",
        files={"files": ("test.txt", b"hello knowledge", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["file_count"] == 1
    saved_files = [f for f in os.listdir(str(tmp_path)) if f.endswith(".txt")]
    assert len(saved_files) == 1, "应有 1 个 txt 文件被保存"


def test_knowledge_rebuild(api_client, monkeypatch):
    class _StubVS:
        def load_document(self):
            return None

        def count(self):
            return 12

    from api.main import app

    # 向注入的容器替换向量库桩（路由经 container.vector_store 取用）
    app.state.container._vector_store = _StubVS()
    resp = api_client.post("/api/knowledge/rebuild")
    assert resp.status_code == 200
    assert resp.json()["chunk_count"] == 12


# ----------------------------------------------------------------- 上传日志清洗（P1-10）
def test_knowledge_upload_filename_sanitized_in_logs(api_client, tmp_path, monkeypatch, caplog):
    """上传文件名含密钥 / 路径时，file_uploaded 与 upload_errors 日志的 formatter 输出零泄漏。"""
    import logging

    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter

    monkeypatch.setattr("api.routes.knowledge.get_abs_path", lambda p: str(tmp_path))
    secret = "sk-UPLOAD-998877665544"
    abs_path = "D:\\secret\\upload\\payload.bin"

    with caplog.at_level(logging.INFO, logger="agent"):
        resp = api_client.post(
            "/api/knowledge/upload",
            files=[
                # 合法文件：文件名仅含裸密钥（无路径分隔符，通过业务校验）
                ("files", (f"{secret}.txt", b"content", "text/plain")),
                # 非法文件：文件名含 Windows 路径（被业务校验拒绝，进 errors 日志）
                ("files", (f"{secret}_{abs_path}.txt", b"bad", "text/plain")),
            ],
        )

    assert resp.status_code == 200
    assert resp.json()["file_count"] == 1, "仅合法文件被保存"

    events = [r.msg.get("event") for r in caplog.records if isinstance(r.msg, dict)]
    assert "file_uploaded" in events, "file_uploaded 事件保留"
    assert "upload_errors" in events, "被拒文件的错误也进日志"

    uploaded = [r.msg for r in caplog.records
                if isinstance(r.msg, dict) and r.msg.get("event") == "file_uploaded"]
    assert uploaded[0]["size"] == len(b"content"), "安全字段（size）保留"

    json_outs = [JsonFormatter().format(r) for r in caplog.records]
    console_outs = [CONSOLE_FORMAT.format(r) for r in caplog.records]
    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert secret not in combined, f"{label}：文件名中的密钥不得泄漏"
        assert abs_path not in combined, f"{label}：文件名中的路径不得泄漏"
        assert abs_path.replace("\\", "\\\\") not in combined, f"{label}：转义路径不得泄漏"
        assert "secret" not in combined.replace("上传日志清洗", ""), f"{label}：路径目录细节不得泄漏"


# ----------------------------------------------------------------- SSE 事件状态机
def test_sse_event_state_machine_chat(api_client):
    """验证 chat SSE 事件包含 message 和 done"""
    resp = api_client.post("/api/chat", json={"query": "你好"})
    assert resp.status_code == 200
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data:"):
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line[5:].strip()))
    types = [e.get("type") for e in events]
    assert "done" in types, "应有 done 事件"
    assert any(t in types for t in ("message", "tool_start", "route")), "应有业务事件"


def test_sse_error_event_state_machine(api_client):
    """验证生成器异常时产生 error 事件"""
    class _Orch:
        def execute(self, *a, **kw):
            def gen():
                yield {"type": "message", "agent": "test", "content": "开始"}
                raise RuntimeError("模拟异常")
            return gen()

    from api.main import app

    app.state.container._orchestrator = _Orch()
    resp = api_client.post("/api/chat", json={"query": "触发异常"})
    assert resp.status_code == 200
    text = resp.text
    assert "error" in text.lower() or '"error"' in text, "异常时应产生 error 事件"


def test_knowledge_upload_multiple_files(api_client, tmp_path, monkeypatch):
    """验证多文件上传"""
    monkeypatch.setattr("api.routes.knowledge.get_abs_path", lambda p: str(tmp_path))
    resp = api_client.post(
        "/api/knowledge/upload",
        files=[
            ("files", ("a.txt", b"content a", "text/plain")),
            ("files", ("b.txt", b"content b", "text/plain")),
        ],
    )
    assert resp.status_code == 200
    assert resp.json()["file_count"] == 2


def test_knowledge_upload_rejects_no_files(api_client):
    """验证不上传文件时返回错误"""
    resp = api_client.post("/api/knowledge/upload", files={})
    assert resp.status_code in (400, 422), "无文件应返回客户端错误"


# ------------------------------------------------------------- 事件循环不被阻塞
@pytest.mark.asyncio
async def test_sync_agent_call_does_not_block_event_loop(monkeypatch):
    """同步 Agent 调用（/chat/sync）跑在线程池，不阻塞同一事件循环上的异步接口。

    场景：/chat/sync 的 Orchestrator 卡在耗时的同步 LLM 调用上，
    此刻并发的 /api/health/live（普通异步接口）必须仍然快速响应。
    """
    import asyncio
    import threading
    import time

    import httpx

    entered = threading.Event()
    release = threading.Event()

    class SlowOrchestrator:
        def execute(self, query, history=None, mode=None):
            entered.set()  # 已进入同步 execute（模拟慢 LLM）
            deadline = time.monotonic() + 5
            while not release.is_set() and time.monotonic() < deadline:
                time.sleep(0.02)
            yield {"type": "message", "agent": "conversation", "content": "慢回复完成"}
            yield {"type": "done", "agent": "conversation", "content": ""}

    from api.container import AppContainer
    from api.main import app

    # 直接经 httpx ASGI 传输请求（无 TestClient 夹具）：手动注入容器 + 慢桩
    container = AppContainer()
    container._orchestrator = SlowOrchestrator()
    monkeypatch.setattr(app.state, "container", container, raising=False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        slow_task = asyncio.create_task(
            client.post("/api/chat/sync", json={"query": "慢请求"})
        )
        try:
            # 等待慢请求确实进入阻塞的 execute（最多 2 秒）
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.02)
            assert entered.is_set(), "慢请求应在 2 秒内进入同步 execute"

            start = time.perf_counter()
            resp = await client.get("/api/health/live")
            elapsed = time.perf_counter() - start

            assert resp.status_code == 200
            assert elapsed < 1.0, (
                f"同步 Agent 调用期间健康检查被阻塞了 {elapsed:.2f}s，"
                "事件循环不应被同步 LLM 调用阻塞"
            )
        finally:
            release.set()
            slow_resp = await asyncio.wait_for(slow_task, timeout=5)

    assert slow_resp.status_code == 200
    assert slow_resp.json()["answer"] == "慢回复完成"


# ----------------------------------------------------------------- 鉴权安全（P1-12）
def test_admin_auth_failure_hides_token(api_client, monkeypatch, caplog):
    """API_TOKEN 设置后：管理接口鉴权失败不泄漏 token（响应与日志均不含）。"""
    import logging

    from api.main import app
    from utils.logger_handler import CONSOLE_FORMAT, JsonFormatter

    real_token = "sk-REALTOKEN-998877665544"
    wrong_token = "sk-WRONGTOKEN-112233445566"
    monkeypatch.setattr("api.security.API_TOKEN", real_token)

    class _StubVS:
        def load_document(self):
            return None

        def count(self):
            return 5

    # 缺少 Authorization 头 → 401
    resp = api_client.post("/api/knowledge/rebuild")
    assert resp.status_code == 401
    assert "detail" in resp.json()
    assert real_token not in resp.text, "响应不得泄漏 token"

    # 错误 token → 403
    with caplog.at_level(logging.WARNING, logger="agent"):
        resp = api_client.post(
            "/api/knowledge/rebuild",
            headers={"Authorization": f"Bearer {wrong_token}"},
        )
    assert resp.status_code == 403
    assert real_token not in resp.text and wrong_token not in resp.text, "403 响应不得泄漏 token"

    auth_events = [r.msg for r in caplog.records
                   if isinstance(r.msg, dict) and r.msg.get("event") in ("auth_failed", "auth_missing")]
    assert auth_events, "应记录鉴权失败事件"
    json_outs = [JsonFormatter().format(r) for r in caplog.records]
    console_outs = [CONSOLE_FORMAT.format(r) for r in caplog.records]
    for label, outs in (("JSON", json_outs), ("控制台", console_outs)):
        combined = "\n".join(outs)
        assert real_token not in combined and wrong_token not in combined, f"{label}：鉴权日志不得泄漏 token"
        assert "Bearer" not in combined, f"{label}：不得记录 Authorization 头"

    # 正确 token → 正常放行（200）
    app.state.container._vector_store = _StubVS()
    resp = api_client.post(
        "/api/knowledge/rebuild",
        headers={"Authorization": f"Bearer {real_token}"},
    )
    assert resp.status_code == 200


# ----------------------------------------------------------------- 配置与启动安全（P1-13）
def test_cors_wildcard_warning_logged(monkeypatch, caplog):
    """CORS_ORIGINS=* 时记录安全警告，提示生产环境配置显式来源。"""
    import logging

    from api.main import _parse_cors_origins

    monkeypatch.setenv("CORS_ORIGINS", "*")
    with caplog.at_level(logging.WARNING, logger="agent"):
        origins = _parse_cors_origins()

    assert origins == ["*"], "通配符解析应保留"
    warnings = [r.msg for r in caplog.records
                if isinstance(r.msg, dict) and r.msg.get("event") == "cors_wildcard_enabled"]
    assert warnings, "应记录 cors_wildcard_enabled 警告"
    assert "生产环境" in warnings[0]["hint"], "警告应给出生产环境建议"


def test_cors_explicit_origins_no_warning(monkeypatch, caplog):
    """显式来源列表：正常解析且不触发通配符警告。"""
    import logging

    from api.main import _parse_cors_origins

    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com, http://localhost:3000")
    with caplog.at_level(logging.WARNING, logger="agent"):
        origins = _parse_cors_origins()

    assert origins == ["https://app.example.com", "http://localhost:3000"]
    assert not any(
        isinstance(r.msg, dict) and r.msg.get("event") == "cors_wildcard_enabled"
        for r in caplog.records
    ), "显式来源不应触发通配符警告"


def test_health_live_does_not_load_model(api_client, monkeypatch):
    """存活探针只表示进程存活：模型工厂被调用即失败，且不依赖容器。"""
    from api.main import app

    def boom(self):
        raise AssertionError("live 探针不得触发模型加载")

    monkeypatch.setattr("model.factory.ChatModelFactory.generator", boom)
    monkeypatch.delattr(app.state, "container", raising=False)

    resp = api_client.get("/api/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_health_ready_chat_model_failure_returns_503(api_client, monkeypatch):
    """就绪探针：chat_model 检查失败时返回 503（不只依赖 vector_store 失败场景）。"""
    from api.container import AppContainer
    from api.main import app

    class BrokenModelContainer(AppContainer):
        @property
        def chat_model(self):
            raise RuntimeError("模型不可用")

    class FakeVectorStore:
        def count(self):
            return 5

    container = BrokenModelContainer()
    container._vector_store = FakeVectorStore()
    monkeypatch.setattr(app.state, "container", container, raising=False)

    resp = api_client.get("/api/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["chat_model"]["ok"] is False
    assert body["checks"]["vector_store"]["ok"] is True, "其他检查不受影响"


# ------------------------------------------ 容器状态与 readiness 统一语义（P1-16）
def test_health_ready_closing_container_returns_503(api_client, monkeypatch):
    """CLOSING 容器：readiness 立即 503，不执行资源检查、不触发懒加载重建。"""
    from api.container import AppContainer, ContainerState
    from api.main import app

    container = AppContainer()
    container._state = ContainerState.CLOSING
    monkeypatch.setattr(app.state, "container", container, raising=False)

    resp = api_client.get("/api/health/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {}, "CLOSING 时不执行任何资源检查"
    assert container._chat_model is None and container._vector_store is None, \
        "探针不得触发懒加载重建"


def test_health_ready_closed_container_returns_503(api_client, monkeypatch):
    """CLOSED 容器：readiness 503，且不因探针重建资源。"""
    from api.container import AppContainer
    from api.main import app

    container = AppContainer()
    container._chat_model = object()  # 模拟曾持有资源
    container.close()
    assert container.closed
    monkeypatch.setattr(app.state, "container", container, raising=False)

    resp = api_client.get("/api/health/ready")

    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"
    assert container._chat_model is None, "CLOSED 容器探针不得重建资源"


def test_health_ready_not_ready_response_has_no_internal_info(api_client, monkeypatch):
    """非 OPEN 容器的 readiness 响应：只有安全结构，无异常原文/内部类型/路径。"""
    from api.container import AppContainer, ContainerState
    from api.main import app

    container = AppContainer()
    container._state = ContainerState.CLOSING
    monkeypatch.setattr(app.state, "container", container, raising=False)

    resp = api_client.get("/api/health/ready")

    assert resp.status_code == 503
    body = resp.json()
    assert set(body) == {"status", "checks"}, "响应只含安全结构字段"
    serialized = str(body)
    for forbidden in ("CLOSING", "ContainerStateError", "Traceback", "D:\\", "/home/", "chat_model"):
        assert forbidden not in serialized, f"readiness 响应不得包含内部信息: {forbidden}"


def test_health_ready_logs_safe_fields_only(api_client, monkeypatch, caplog):
    """readiness 拒绝非 OPEN 容器时记 container_not_ready：字段恰为安全四元组。"""
    import logging

    from api.container import AppContainer
    from api.main import app

    container = AppContainer()
    container.close()
    monkeypatch.setattr(app.state, "container", container, raising=False)

    with caplog.at_level(logging.WARNING, logger="agent"):
        resp = api_client.get("/api/health/ready")

    assert resp.status_code == 503
    records = [r.msg for r in caplog.records
               if isinstance(r.msg, dict) and r.msg.get("event") == "container_not_ready"]
    assert len(records) == 1, "应记录一条 container_not_ready 日志"
    logged = records[0]
    assert set(logged) == {"event", "state", "request_id", "error_code"}, \
        "日志只允许安全字段，不得记录异常原文"
    assert logged["state"] == "closed"
    assert logged["error_code"] == "CONTAINER_NOT_READY"
    assert logged["request_id"], "request_id 应随请求上下文记录"
    # 任何字段值都不携带异常原文或内部细节
    serialized = str(logged)
    assert "Traceback" not in serialized and "RuntimeError" not in serialized


def test_health_ready_unmounted_container_returns_503_and_logs(api_client, monkeypatch, caplog):
    """未挂载容器：readiness 503 且 get_app_container 记 container_not_ready 日志。"""
    import logging

    from api.main import app

    monkeypatch.delattr(app.state, "container", raising=False)
    with caplog.at_level(logging.WARNING, logger="agent"):
        resp = api_client.get("/api/health/ready")

    assert resp.status_code == 503, "就绪探针在无容器时应返回 503"
    assert "容器" in resp.json()["detail"], "应给出明确原因"
    records = [r.msg for r in caplog.records
               if isinstance(r.msg, dict) and r.msg.get("event") == "container_not_ready"]
    assert len(records) == 1, "未挂载同样应记录 container_not_ready"
    assert set(records[0]) == {"event", "state", "request_id", "error_code"}
    assert records[0]["state"] == "unmounted"


# ------------------------------------------------- lifespan 启动校验接入（P1-13.1）
@pytest.mark.asyncio
async def test_lifespan_startup_success_mounts_container(monkeypatch):
    """校验通过：容器挂载到 app.state，退出 lifespan 后容器被关闭。"""
    from fastapi import FastAPI

    from api import main as main_mod
    from api.container import AppContainer

    monkeypatch.setattr(main_mod, "validate_startup", lambda: None)
    app = FastAPI()

    async with main_mod.lifespan(app):
        assert isinstance(app.state.container, AppContainer), "校验通过后应挂载容器"
        assert not app.state.container.closed
    assert app.state.container.closed, "退出 lifespan 应关闭容器"


@pytest.mark.asyncio
async def test_lifespan_validation_failure_blocks_startup(monkeypatch, caplog):
    """校验失败：日志统一 safe_exception_fields 形态，异常原样抛出，零容器构造。"""
    import logging

    from fastapi import FastAPI

    from api import main as main_mod
    from utils.config_validator import ConfigValidationError

    # 消息携带密钥与 Windows / Unix 绝对路径形态，验证日志侧统一脱敏
    sentinel_key = "sk-abc123def456ghi789"
    win_path = "D:\\keys\\rag.yml"
    unix_path = "/home/ops/.env"
    failure = ConfigValidationError(f"配置校验失败: {sentinel_key} {win_path} {unix_path}")

    def boom():
        raise failure

    def forbidden_container():
        raise AssertionError("校验失败时不得构造新容器")

    monkeypatch.setattr(main_mod, "validate_startup", boom)
    monkeypatch.setattr(main_mod, "AppContainer", forbidden_container)

    app = FastAPI()
    with caplog.at_level(logging.ERROR, logger="agent"), pytest.raises(ConfigValidationError) as excinfo:
        async with main_mod.lifespan(app):
            pass

    # 异常原样向外抛出：同一实例、消息未被日志逻辑改写
    assert excinfo.value is failure
    assert sentinel_key in str(excinfo.value)

    # 日志记录恰为 event / error_type / error_msg 三字段（safe_exception_fields 统一形态）
    records = [r for r in caplog.records
               if isinstance(r.msg, dict) and r.msg.get("event") == "startup_config_validation_failed"]
    assert len(records) == 1, "应记录一条启动校验失败日志"
    logged = records[0].msg
    assert set(logged) == {"event", "error_type", "error_msg"}, "日志只含统一异常摘要字段"
    assert logged["error_type"] == "ConfigValidationError"

    error_msg = logged["error_msg"]
    # 密钥被脱敏（原值不出现，脱敏占位存在）
    assert sentinel_key not in error_msg
    assert "***REDACTED***" in error_msg
    # Windows / Unix 绝对路径不出现
    assert win_path not in error_msg and unix_path not in error_msg
    assert "<PATH_REDACTED>" in error_msg
    # 无 traceback / exc_info：异常未随日志附带调用栈
    assert records[0].exc_info is None
    assert "Traceback" not in error_msg

    # 失败时不得构造、不得挂载容器
    assert getattr(app.state, "container", None) is None, "失败时不得挂载容器"


@pytest.mark.asyncio
async def test_lifespan_calls_validate_startup_once(monkeypatch):
    """校验恰调用一次：启动期一次，运行与关闭阶段均不重复执行。"""
    from fastapi import FastAPI

    from api import main as main_mod

    calls = []
    monkeypatch.setattr(main_mod, "validate_startup", lambda: calls.append(1))

    app = FastAPI()
    async with main_mod.lifespan(app):
        assert len(calls) == 1, "启动阶段应校验一次"
    assert len(calls) == 1, "关闭阶段不得再次校验"


@pytest.mark.asyncio
async def test_lifespan_startup_loads_no_heavy_resources(monkeypatch):
    """启动零重型资源：模型 / Embedding / Chroma / RAG / Orchestrator 全部保持懒加载。"""
    from fastapi import FastAPI

    from api import main as main_mod

    def boom(self=None):
        raise AssertionError("启动阶段不得加载模型或 Embedding")

    monkeypatch.setattr(main_mod, "validate_startup", lambda: None)
    monkeypatch.setattr("model.factory.ChatModelFactory.generator", boom)
    monkeypatch.setattr("model.factory.EmbeddingsFactory.generator", boom)

    app = FastAPI()
    async with main_mod.lifespan(app):
        container = app.state.container
        assert container._chat_model is None, "启动不得加载 Chat 模型"
        assert container._embedding_model is None, "启动不得加载 Embedding"
        assert container._vector_store is None, "启动不得加载 Chroma 向量库"
        assert container._rag_service is None, "启动不得构建 RAG 服务"
        assert container._orchestrator is None, "启动不得构建 Orchestrator"
