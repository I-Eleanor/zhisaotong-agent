"""P1-2 生命周期测试：单例双检锁 + AppContainer 依赖注入。

核心验证：
- 并发首次初始化时，chat 模型 / orchestrator / 诊断 Agent 只创建一次；
- AppContainer 全量持有并复用资源（模型/Embedding/向量库/RAG/orchestrator），
  重复与并发访问均只初始化一次；两个容器互不共享实例；
- orchestrator 由容器管理的依赖组装（模型注入 Agent，RAG 工具绑定容器 rag_service）；
- close() 幂等且只释放容器自己的资源，不触碰全局单例；
- API 路由经 Depends 使用注入容器（模块级 getter 抛异常仍正常）；
  无容器时返回明确 503，不隐式创建资源；
- lifespan 每次启动新建容器、关闭只关该实例、启动不加载重型资源。
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient

from utils.exceptions import ContainerStateError


def _race_workers(worker, count: int = 4, hold_seconds: float = 0.2, gate: threading.Event | None = None) -> list:
    """并发执行 worker 若干次（各占一个线程），返回结果列表。

    hold_seconds：并发窗口时长；gate：可选，在 join 前放行卡在构造里的首个线程。
    """
    results: list = []
    lock = threading.Lock()

    def run():
        result = worker()
        with lock:
            results.append(result)

    threads = [threading.Thread(target=run) for _ in range(count)]
    for t in threads:
        t.start()
    time.sleep(hold_seconds)
    if gate is not None:
        gate.set()
    for t in threads:
        t.join(timeout=5)
    return results


# ----------------------------------------------------------------- 模型工厂并发初始化
def test_concurrent_chat_model_created_once(monkeypatch):
    """并发首次调用 get_chat_model() 应只创建一次模型，且返回同一实例。"""
    import model.factory as mf

    calls: list[int] = []
    gate = threading.Event()

    class SlowChatFactory:
        def generator(self):
            calls.append(1)
            gate.wait(timeout=3)  # 放慢首次创建，制造并发窗口
            return object()

    monkeypatch.setattr(mf, "ChatModelFactory", SlowChatFactory)
    mf.reset_models()
    try:
        results = _race_workers(mf.get_chat_model, gate=gate)

        assert len(results) == 4, "4 个线程都应拿到模型实例"
        assert len(calls) == 1, (
            f"并发首次初始化应只创建一次 chat 模型，实际创建了 {len(calls)} 次"
        )
        assert all(r is results[0] for r in results), "所有线程应拿到同一个模型实例"
    finally:
        mf.reset_models()


def test_concurrent_embed_model_created_once(monkeypatch):
    """并发首次调用 get_embed_model() 应只创建一次。"""
    import model.factory as mf

    calls: list[int] = []

    class SlowEmbedFactory:
        def generator(self):
            calls.append(1)
            time.sleep(0.2)
            return object()

    monkeypatch.setattr(mf, "EmbeddingsFactory", SlowEmbedFactory)
    mf.reset_models()
    try:
        results = _race_workers(mf.get_embed_model)
        assert len(calls) == 1, "并发首次初始化应只创建一次 embedding 模型"
        assert all(r is results[0] for r in results)
    finally:
        mf.reset_models()


# ----------------------------------------------------------------- Orchestrator 并发初始化
def test_concurrent_orchestrator_created_once(monkeypatch):
    """并发首次调用 get_orchestrator() 应只构建一次 Agent。"""
    import agent.orchestrator as orch_mod

    calls: list[int] = []
    gate = threading.Event()

    class SlowOrchestrator:
        def __init__(self, *args, **kwargs):
            calls.append(1)
            gate.wait(timeout=3)

    monkeypatch.setattr(orch_mod, "Orchestrator", SlowOrchestrator)
    orch_mod.reset_orchestrator()
    try:
        results = _race_workers(orch_mod.get_orchestrator, gate=gate)

        assert len(calls) == 1, (
            f"并发首次初始化应只构建一次 Orchestrator，实际构建了 {len(calls)} 次"
        )
        assert all(r is results[0] for r in results)
    finally:
        orch_mod.reset_orchestrator()


# ----------------------------------------------------------------- 诊断 Agent 并发初始化
def test_concurrent_diagnostic_agent_created_once(monkeypatch):
    """并发首次调用 get_diagnostic_agent() 应只编译一次图。"""
    import agent.diagnostic.service as svc

    calls: list[int] = []
    gate = threading.Event()

    class SlowAgent:
        def __init__(self, *args, **kwargs):
            calls.append(1)
            gate.wait(timeout=3)

    monkeypatch.setattr(svc, "DiagnosticAgent", SlowAgent)
    svc.reset_diagnostic_agent()
    try:
        results = _race_workers(svc.get_diagnostic_agent, gate=gate)

        assert len(calls) == 1, (
            f"并发首次初始化应只编译一次诊断图，实际编译了 {len(calls)} 次"
        )
        assert all(r is results[0] for r in results)
    finally:
        svc.reset_diagnostic_agent()


# ----------------------------------------------------------------- AppContainer 行为
def _patch_container_resources(monkeypatch):
    """把容器懒加载用到的工厂替换为计数桩（不加载真实模型/Chroma/重排器）。

    返回 (counters, fakes)：counters 记录各资源创建次数，fakes 提供桩类引用。
    """
    import agent.conversation_agent as ca_mod
    import agent.diagnostic.service as ds_mod
    import agent.orchestrator as orch_mod
    import model.factory as mf
    from rag import rag_service as rs_mod
    from rag import vector_store as vs_mod

    counters = {"chat": 0, "embed": 0, "vs": 0, "rag": 0, "conv": 0, "diag": 0, "orch": 0}

    class FakeChat:
        pass

    class FakeEmbed:
        pass

    class FakeVS:
        def __init__(self, embedding_function=None, **kwargs):
            counters["vs"] += 1
            self.embedding_function = embedding_function

    class FakeRag:
        def __init__(self, vector_store=None, model=None):
            counters["rag"] += 1
            self.vector_store = vector_store
            self.model = model

        def rag_summarize(self, query, source_files=None):
            return "RAG结果"

    class FakeConvAgent:
        def __init__(self, model=None, tools=None, middleware=None, system_prompt=None):
            counters["conv"] += 1
            self.model = model
            self.tools = tools

    class FakeDiagAgent:
        def __init__(self, parser=None, tool_router=None, model=None):
            counters["diag"] += 1
            self.parser = parser
            self.tool_router = tool_router
            self.model = model

    class FakeOrchestrator:
        def __init__(self, conversation_agent=None, diagnostic_agent=None):
            counters["orch"] += 1
            self.conversation_agent = conversation_agent
            self.diagnostic_agent = diagnostic_agent

    def _fake_chat(self=None):
        counters["chat"] += 1
        return FakeChat()

    def _fake_embed(self=None):
        counters["embed"] += 1
        return FakeEmbed()

    monkeypatch.setattr(mf.ChatModelFactory, "generator", _fake_chat)
    monkeypatch.setattr(mf.EmbeddingsFactory, "generator", _fake_embed)
    monkeypatch.setattr(vs_mod, "VectorStoreService", FakeVS)
    monkeypatch.setattr(rs_mod, "RagSummarizeService", FakeRag)
    monkeypatch.setattr(ca_mod, "ConversationAgent", FakeConvAgent)
    monkeypatch.setattr(ds_mod, "DiagnosticAgent", FakeDiagAgent)
    monkeypatch.setattr(orch_mod, "Orchestrator", FakeOrchestrator)

    fakes = {
        "chat": FakeChat, "embed": FakeEmbed, "vs": FakeVS, "rag": FakeRag,
        "conv": FakeConvAgent, "diag": FakeDiagAgent, "orch": FakeOrchestrator,
    }
    return counters, fakes


def test_container_resources_created_once_and_cached(monkeypatch):
    """同一容器重复访问每个依赖：全部只初始化一次并返回同一实例。"""
    from api.container import AppContainer

    counters, _ = _patch_container_resources(monkeypatch)
    container = AppContainer()

    assert container.chat_model is container.chat_model
    assert container.embedding_model is container.embedding_model
    assert container.vector_store is container.vector_store
    assert container.rag_service is container.rag_service
    assert container.orchestrator is container.orchestrator

    assert counters == {"chat": 1, "embed": 1, "vs": 1, "rag": 1,
                        "conv": 1, "diag": 1, "orch": 1}, "每个资源只创建一次"


def test_container_concurrent_access_initializes_once(monkeypatch):
    """并发首次访问 orchestrator（级联创建模型/Embedding/向量库/RAG）：每个资源只初始化一次。"""
    from api.container import AppContainer

    counters, _ = _patch_container_resources(monkeypatch)
    container = AppContainer()

    results = _race_workers(lambda: container.orchestrator, count=6, hold_seconds=0.05)

    assert len(results) == 6
    assert all(r is results[0] for r in results), "并发访问应拿到同一个 orchestrator"
    assert counters["orch"] == 1, f"orchestrator 只应创建一次，实际 {counters['orch']} 次"
    assert counters["chat"] == 1 and counters["embed"] == 1
    assert counters["vs"] == 1 and counters["rag"] == 1


def test_two_containers_do_not_share_resources(monkeypatch):
    """两个独立容器各自创建资源实例，互不共享。"""
    from api.container import AppContainer

    _patch_container_resources(monkeypatch)
    c1, c2 = AppContainer(), AppContainer()

    assert c1.chat_model is not c2.chat_model
    assert c1.embedding_model is not c2.embedding_model
    assert c1.vector_store is not c2.vector_store
    assert c1.rag_service is not c2.rag_service
    assert c1.orchestrator is not c2.orchestrator


def test_container_orchestrator_built_from_managed_deps(monkeypatch):
    """orchestrator 必须用容器管理的依赖组装：模型注入两个 Agent，RAG 工具绑定容器 rag_service。"""
    from api.container import AppContainer

    counters, fakes = _patch_container_resources(monkeypatch)
    container = AppContainer()

    orch = container.orchestrator
    assert isinstance(orch, fakes["orch"]), "Orchestrator 应由容器构造"
    assert isinstance(orch.conversation_agent, fakes["conv"])
    assert isinstance(orch.diagnostic_agent, fakes["diag"])

    # 模型注入：两个 Agent 用的都是容器持有的 chat_model
    assert orch.conversation_agent.model is container.chat_model
    assert orch.diagnostic_agent.model is container.chat_model

    # RAG 工具绑定：对话 Agent 的 rag 工具与诊断 Agent 的 retrieve_knowledge
    # 都应调用容器管理的 rag_service（经 FakeRag 返回固定文本）
    rag_tool = orch.conversation_agent.tools[0]
    assert rag_tool.name == "rag_summarize"
    assert rag_tool.invoke({"query": "滤网"}) == "RAG结果"

    from agent.diagnostic.schemas import DiagnosticStep

    step = DiagnosticStep(description="查资料", tool="retrieve_knowledge", arguments={"query": "滤网"})
    result = orch.diagnostic_agent.tool_router.execute(step, user_query="扫地机不工作")
    assert result.success is True
    assert result.content == "RAG结果", "诊断工具应走容器管理的知识库 Agent"


def test_container_close_is_idempotent(monkeypatch):
    """close() 连续调用两次不抛异常。"""
    from api.container import AppContainer

    _patch_container_resources(monkeypatch)
    container = AppContainer()
    container.close()
    container.close()  # 第二次 close：幂等，不抛异常
    assert container.closed is True


def test_container_close_releases_and_reinitializes(monkeypatch):
    """close() 清空资源缓存；此后再次访问按懒加载重新初始化。"""
    from api.container import AppContainer

    counters, _ = _patch_container_resources(monkeypatch)
    container = AppContainer()

    rag1 = container.rag_service
    container.close()
    assert container.closed

    rag2 = container.rag_service
    assert rag2 is not rag1, "close() 释放后再次访问应重新初始化"
    assert counters["rag"] == 2


def test_container_close_does_not_touch_global_singletons():
    """close() 只释放容器自己的资源，不得重置模块级全局单例（可能被旧入口持有）。"""
    import agent.diagnostic.service as ds_mod
    import agent.orchestrator as orch_mod
    import model.factory as mf
    from api.container import AppContainer

    class _Fake:
        pass

    mf._chat_model = _Fake()
    orch_mod._orchestrator = _Fake()
    ds_mod._diagnostic_agent = _Fake()
    try:
        container = AppContainer()
        container.close()
        assert mf._chat_model is not None, "close() 不得调用 reset_models()"
        assert orch_mod._orchestrator is not None, "close() 不得调用 reset_orchestrator()"
        assert ds_mod._diagnostic_agent is not None, "close() 不得调用 reset_diagnostic_agent()"
    finally:
        mf.reset_models()
        orch_mod.reset_orchestrator()
        ds_mod.reset_diagnostic_agent()


# ----------------------------------------------------------------- close() 释放语义（P1-14）
class _ClosableResource:
    """带 close() 的资源桩：把自身名字按关闭顺序记入共享列表。"""

    def __init__(self, name: str, closed_log: list[str]):
        self.name = name
        self._closed_log = closed_log

    def close(self) -> None:
        self._closed_log.append(self.name)


def test_container_close_releases_in_reverse_dependency_order():
    """五类资源按依赖逆序关闭：orchestrator → rag → 向量库 → Embedding → chat。"""
    from api.container import AppContainer

    closed: list[str] = []
    container = AppContainer()
    container._orchestrator = _ClosableResource("orchestrator", closed)
    container._rag_service = _ClosableResource("rag_service", closed)
    container._vector_store = _ClosableResource("vector_store", closed)
    container._embedding_model = _ClosableResource("embedding_model", closed)
    container._chat_model = _ClosableResource("chat_model", closed)

    container.close()

    assert closed == ["orchestrator", "rag_service", "vector_store",
                      "embedding_model", "chat_model"], "必须依赖逆序释放"


def test_container_close_skips_resources_without_close():
    """资源缺少 close()：不抛异常，引用仍被清空。"""
    from api.container import AppContainer

    container = AppContainer()
    container._orchestrator = object()
    container._rag_service = object()
    container._vector_store = object()
    container._embedding_model = object()
    container._chat_model = object()

    container.close()  # 无 close() 的资源：静默跳过，不抛异常

    assert container.closed is True
    assert container._orchestrator is None
    assert container._rag_service is None
    assert container._vector_store is None
    assert container._embedding_model is None
    assert container._chat_model is None, "引用清空后才能重新懒加载"


def test_container_close_continues_after_resource_failure(caplog):
    """中途资源关闭抛异常：记结构化日志、继续释放剩余资源、不向外抛异常。"""
    import logging

    from api.container import AppContainer

    closed: list[str] = []

    class _BrokenClose:
        def close(self):
            raise RuntimeError("关闭失败")

    container = AppContainer()
    container._orchestrator = _ClosableResource("orchestrator", closed)
    container._rag_service = _BrokenClose()  # 逆序第 2 项失败
    container._vector_store = _ClosableResource("vector_store", closed)
    container._embedding_model = _ClosableResource("embedding_model", closed)
    container._chat_model = _ClosableResource("chat_model", closed)

    with caplog.at_level(logging.WARNING, logger="agent"):
        container.close()  # 不得向外抛关闭异常（避免遮蔽应用退出）

    assert closed == ["orchestrator", "vector_store", "embedding_model", "chat_model"], \
        "失败后续资源仍应释放"
    failures = [r.msg for r in caplog.records
                if isinstance(r.msg, dict) and r.msg.get("event") == "container_resource_close_failed"]
    assert len(failures) == 1, "应记录一条资源关闭失败日志"
    assert set(failures[0]) == {"event", "resource", "error_type", "error_msg"}, \
        "失败日志为 safe_exception_fields 统一形态 + resource 定位字段"
    assert failures[0]["resource"] == "rag_service"
    assert failures[0]["error_type"] == "RuntimeError"
    # 失败资源同样被清空引用
    assert container._rag_service is None and container._chat_model is None


def test_container_close_repeated_does_not_reclose_resources():
    """重复 close()：同一资源只关闭一次。"""
    from api.container import AppContainer

    closed: list[str] = []
    container = AppContainer()
    container._vector_store = _ClosableResource("vector_store", closed)
    container._chat_model = _ClosableResource("chat_model", closed)

    container.close()
    container.close()  # 第二次：引用已清空，不得重复关闭

    assert closed == ["vector_store", "chat_model"], "每个资源恰关闭一次"
    assert container.closed is True


def test_container_close_then_rebuild_and_close_again(monkeypatch):
    """关闭后资源可重建（引用清空、重新懒加载），重建后再次 close 能释放新资源。"""
    counters, _ = _patch_container_resources(monkeypatch)
    from api.container import AppContainer

    container = AppContainer()
    chat1 = container.chat_model
    container.close()
    assert container.closed is True
    assert container._chat_model is None

    chat2 = container.chat_model  # 重新懒加载
    assert chat2 is not chat1
    assert counters["chat"] == 2
    assert container.closed is False, "重新持有资源后容器复活"

    reopened: list[int] = []

    def _track_close():
        reopened.append(1)

    chat2.close = _track_close  # 给重建实例装上 close，验证复活后仍可释放
    container.close()
    assert reopened == [1], "重建的资源再次 close() 应被释放"


def test_container_close_does_not_affect_other_containers():
    """两个容器互不影响：关闭 c1 不触碰 c2 的资源与状态。"""
    from api.container import AppContainer

    closed: list[str] = []
    c1, c2 = AppContainer(), AppContainer()
    c1._chat_model = _ClosableResource("c1-chat", closed)
    c2._chat_model = _ClosableResource("c2-chat", closed)

    c1.close()

    assert closed == ["c1-chat"], "只关闭 c1 自己的资源"
    assert c1.closed is True and c2.closed is False
    assert c2._chat_model is not None and c2._chat_model.name == "c2-chat", "c2 资源不受影响"


# ------------------------------------------------- close() 并发语义（P1-15 竞态修复）
class _GateCloseResource:
    """close() 进入即置 entered、放行前阻塞的资源桩：撑开 CLOSING 窗口。"""

    def __init__(self, entered: threading.Event, gate: threading.Event, closed_log: list[str], name: str):
        self.entered = entered
        self.gate = gate
        self.closed_log = closed_log
        self.name = name

    def close(self) -> None:
        self.entered.set()  # 释放阶段已开始（此时状态必为 CLOSING）
        self.gate.wait(timeout=5)
        self.closed_log.append(self.name)


def test_access_during_closing_raises_and_creates_nothing(monkeypatch):
    """close() 释放阶段并发访问 chat_model：抛 ContainerStateError，不创建任何新资源。"""
    import model.factory as mf
    from api.container import AppContainer, ContainerState

    creations: list[int] = []

    def _counting_factory(self=None):
        creations.append(1)
        return object()

    monkeypatch.setattr(mf.ChatModelFactory, "generator", _counting_factory)

    entered, gate = threading.Event(), threading.Event()
    closed: list[str] = []
    container = AppContainer()
    container._chat_model = _GateCloseResource(entered, gate, closed, "chat-old")

    closer = threading.Thread(target=container.close)
    closer.start()
    entered.wait(timeout=5)  # 确定进入释放阶段（旧实现竞态窗口）

    assert container.state is ContainerState.CLOSING
    with pytest.raises(ContainerStateError) as excinfo:
        _ = container.chat_model  # CLOSING 期间：拒绝创建新资源
    err = excinfo.value
    # 项目统一异常：稳定错误码 + 固定安全提示（不含状态 / 资源名 / 线程信息）
    assert err.error_code == "CONTAINER_NOT_READY"
    assert err.safe_message == "服务尚未就绪，请稍后重试"
    assert "CLOSING" not in err.safe_message and "chat_model" not in err.safe_message
    assert creations == [], "CLOSING 期间不得产生游离于生命周期之外的新资源"

    gate.set()
    closer.join(timeout=5)
    assert closed == ["chat-old"]
    assert container.state is ContainerState.CLOSED
    assert container.closed is True


def test_resource_created_under_close_contention_is_managed(monkeypatch):
    """close() 与懒加载争锁：close 等待创建完成后快照接管新资源并释放，不泄漏。"""
    import model.factory as mf
    from api.container import AppContainer, ContainerState

    closed: list[str] = []
    entered, gate = threading.Event(), threading.Event()

    def _slow_factory(self=None):
        entered.set()  # 创建开始（此时已持有容器锁）
        gate.wait(timeout=5)
        return _ClosableResource("chat-created", closed)

    monkeypatch.setattr(mf.ChatModelFactory, "generator", _slow_factory)

    container = AppContainer()
    loaded: list[object] = []

    def _load():
        loaded.append(container.chat_model)

    loader = threading.Thread(target=_load)
    loader.start()
    entered.wait(timeout=5)  # loader 持锁阻塞在工厂内
    closer = threading.Thread(target=container.close)
    closer.start()
    time.sleep(0.1)  # closer 阻塞在锁上：拿不到锁不得提前置 CLOSING
    assert container.state is ContainerState.OPEN, "争锁期间状态不得被提前破坏"

    gate.set()  # loader 完成创建 → closer 接管锁 → 快照含新资源
    loader.join(timeout=5)
    closer.join(timeout=5)

    assert closed == ["chat-created"], "争锁期间创建的资源必须被 close 接管释放"
    assert container.state is ContainerState.CLOSED
    assert container._chat_model is None


def test_after_close_resources_recreate_per_design(monkeypatch):
    """CLOSED 后资源访问按设计重新懒加载：新实例、状态回到 OPEN。"""
    import model.factory as mf
    from api.container import AppContainer, ContainerState

    monkeypatch.setattr(mf.ChatModelFactory, "generator", lambda self=None: object())

    container = AppContainer()
    first = container.chat_model
    container.close()
    assert container.state is ContainerState.CLOSED

    second = container.chat_model  # 关闭完成（非关闭中间）：允许重建
    assert second is not first
    assert container.state is ContainerState.OPEN
    assert container.closed is False


def test_close_failure_state_eventually_consistent(monkeypatch):
    """资源关闭抛异常：状态仍收敛到 CLOSED、引用清空、不外抛，且后续可重建。"""
    import model.factory as mf
    from api.container import AppContainer, ContainerState

    class _BrokenClose:
        def close(self) -> None:
            raise RuntimeError("关闭失败")

    monkeypatch.setattr(mf.ChatModelFactory, "generator", lambda self=None: object())

    container = AppContainer()
    container._chat_model = _BrokenClose()
    container.close()  # 不外抛（finally 保证状态收敛）

    assert container.state is ContainerState.CLOSED, "关闭失败后状态必须最终一致"
    assert container._chat_model is None

    rebuilt = container.chat_model  # 失败收敛后仍可正常重建
    assert rebuilt is not None
    assert container.state is ContainerState.OPEN


def test_concurrent_close_releases_each_resource_once():
    """多线程并发 close()：每个资源恰释放一次，全部返回后状态为 CLOSED。"""
    from api.container import AppContainer, ContainerState

    entered, gate = threading.Event(), threading.Event()
    closed: list[str] = []
    container = AppContainer()
    container._vector_store = _GateCloseResource(entered, gate, closed, "vs")

    # 4 个线程并发 close：第 1 个进入释放阶段阻塞在 gate，其余 3 个在
    # Condition 上等待；gate 放行后第 1 个完成 → CLOSED → 唤醒其余直接返回
    _race_workers(container.close, count=4, hold_seconds=0.2, gate=gate)

    assert closed == ["vs"], "并发 close 不得重复释放同一资源"
    assert container.state is ContainerState.CLOSED
    assert container.closed is True


# ----------------------------------------------------------------- API 依赖注入
def test_api_routes_use_injected_container_not_global_getters(api_client, monkeypatch):
    """模块级全局 getter 全部抛异常时，API 仍正常工作 → 路由确实使用注入容器。"""
    import agent.orchestrator as orch_mod
    import model.factory as mf
    from api.main import app

    def _boom(*args, **kwargs):
        raise AssertionError("API 请求路径不得调用模块级全局 getter")

    monkeypatch.setattr(orch_mod, "get_orchestrator", _boom)
    monkeypatch.setattr(mf, "get_chat_model", _boom)

    # 对话（SSE）：走容器内桩 orchestrator
    resp = api_client.post("/api/chat", json={"query": "你好"})
    assert resp.status_code == 200
    assert "event: message" in resp.text

    # 知识查询：走容器 rag_service
    class _StubRag:
        def rag_with_sources(self, query):
            return {"answer": "A", "sources": [], "confidence": 0.9}

    app.state.container._rag_service = _StubRag()
    resp = api_client.post("/api/knowledge/query", json={"query": "q"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "A"

    # 就绪探针：走容器 vector_store / chat_model（工厂直建，不经全局 getter）
    class _StubVS:
        def count(self):
            return 5

    app.state.container._vector_store = _StubVS()
    resp = api_client.get("/api/health/ready")
    assert resp.status_code == 200
    assert resp.json()["checks"]["vector_store"]["chunk_count"] == 5


def test_api_returns_503_when_container_missing(monkeypatch):
    """未进入 lifespan 且未注入容器：返回明确 503，不隐式创建任何资源。"""

    from api import container as container_mod
    from api.main import app

    monkeypatch.delattr(app.state, "container", raising=False)
    client = TestClient(app)

    resp = client.get("/api/health/ready")
    assert resp.status_code == 503, "就绪探针在无容器时应返回 503"
    assert "容器" in resp.json()["detail"], "应给出明确原因"

    resp2 = client.post("/api/chat", json={"query": "你好"})
    assert resp2.status_code == 503, "业务路由在无容器时同样 503"

    # 生产代码不得保留进程级全局容器回退入口
    assert not hasattr(container_mod, "get_container"), "不得回退到进程全局容器"


# ----------------------------------------------------------------- lifespan 集成
def test_lifespan_creates_and_closes_container():
    """TestClient 上下文管理器触发 lifespan：启动时挂载容器，关闭时释放。"""

    from api.main import app

    try:
        with TestClient(app) as client:
            resp = client.get("/api/health/live")
            assert resp.status_code == 200
            container = getattr(app.state, "container", None)
            assert container is not None, "lifespan 启动时应把容器挂载到 app.state"
            assert container.closed is False

        assert app.state.container.closed is True, "应用关闭时容器应被释放"
    finally:
        del app.state.container  # 清理挂载，避免污染后续测试


def test_lifespan_creates_new_container_each_startup():
    """每次启动都创建新的 AppContainer（不复用旧实例，旧实例已关闭）。"""

    from api.main import app

    try:
        with TestClient(app):
            first = app.state.container
        with TestClient(app):
            second = app.state.container
            assert second is not first, "每次启动应创建新容器"
        assert first.closed and second.closed
    finally:
        del app.state.container


def test_lifespan_closes_only_container_it_created():
    """运行期间 app.state.container 被替换：退出 lifespan 只关闭 lifespan 创建的原始实例。

    lifespan 保存局部引用，关闭时不得关错容器（替换进去的实例不属于
    本 lifespan 所有权，由替换方自行管理生命周期）。
    """
    from api.container import AppContainer
    from api.main import app

    replacement = AppContainer()
    try:
        with TestClient(app):
            original = app.state.container
            app.state.container = replacement  # 模拟测试 / 重载逻辑替换容器
        assert original.closed is True, "lifespan 应关闭自己创建的原始容器"
        assert replacement.closed is False, "替换进来的容器不应被 lifespan 关闭"
    finally:
        replacement.close()
        del app.state.container  # 清理挂载，避免污染后续测试


def test_lifespan_startup_does_not_load_resources(monkeypatch):
    """启动阶段不得初始化模型 / Embedding / 向量库（容器全懒加载）。"""
    import model.factory as mf
    from rag import vector_store as vs_mod

    def _boom(*args, **kwargs):
        raise AssertionError("启动阶段不得加载模型 / Embedding / 向量库")

    monkeypatch.setattr(mf.ChatModelFactory, "generator", _boom)
    monkeypatch.setattr(mf.EmbeddingsFactory, "generator", _boom)
    monkeypatch.setattr(vs_mod, "VectorStoreService", _boom)


    from api.main import app

    try:
        with TestClient(app) as client:
            resp = client.get("/api/health/live")
            assert resp.status_code == 200, "存活探针不依赖任何重型资源"
    finally:
        del app.state.container
