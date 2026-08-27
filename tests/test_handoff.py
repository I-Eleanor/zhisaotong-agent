"""人工转接（handoff）功能测试。

覆盖：
1. 关键词触发：Orchestrator 拦截"转人工/人工客服/联系人工"，优先级高于 mode 路由；
2. 事件兼容：非转人工查询原样透传 conversation / diagnostic 事件流；
3. SSE 桥：handoff_suggested 在有界队列满时仍必达（控制通道，不静默丢弃）；
4. /api/chat 与 /api/chat/sync 全链路透传，sync 返回 need_handoff；
5. SQLite 工单存储：CRUD / 状态 / 归属校验；
6. 工单接口：用户创建/查询、管理端列表/详情/回复、ADMIN_TOKEN 鉴权。
"""
import asyncio
import json

import pytest

from agent.events import make_event
from agent.orchestrator import Orchestrator, match_handoff_keyword


# ===================================================================== 关键词
def test_match_handoff_keyword_hits():
    for q in ("帮我转人工", "请找人工客服", "我要联系人工！", "转人工"):
        assert match_handoff_keyword(q), f"{q} 应命中转人工关键词"


def test_match_handoff_keyword_misses():
    for q in ("人工智能是什么", "扫地机不工作", "怎么清理滤网", ""):
        assert match_handoff_keyword(q) is None, f"{q} 不应误触发"


class _NoGoAgent:
    """任何调用都视为路由错误的哨兵 Agent。"""

    def stream(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("conversation agent 不应被调用")

    def run(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("diagnostic agent 不应被调用")


def _handoff_orchestrator() -> Orchestrator:
    return Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent())


def test_orchestrator_keyword_intercepts_before_routing():
    events = list(_handoff_orchestrator().execute("你好，我要转人工"))
    types = [e["type"] for e in events]
    assert types == ["handoff_suggested", "done"]
    assert events[0]["agent"] == "orchestrator"
    assert events[0]["content"], "应携带对用户可见的安全提示"
    assert events[0]["data"]["reason"] == "user_request"
    assert events[0]["data"]["keyword"] == "转人工"


@pytest.mark.parametrize("query,keyword", [
    ("麻烦人工客服解答", "人工客服"),
    ("请帮忙联系人工", "联系人工"),
])
def test_orchestrator_handover_overrides_explicit_mode(query, keyword):
    """显式 mode=diagnostic 也被用户显式转人工拦截。"""
    events = list(_handoff_orchestrator().execute(query, mode="diagnostic"))
    assert [e["type"] for e in events] == ["handoff_suggested", "done"]
    assert events[0]["data"]["keyword"] == keyword


# ===================================================================== 事件兼容
class _EchoConversationAgent:
    def stream(self, query, history=None):
        yield {"type": "message", "agent": "conversation", "content": f"echo:{query}"}
        yield {"type": "done", "agent": "conversation"}


class _EchoDiagnosticAgent:
    def run(self, query):
        yield {"type": "report", "agent": "diagnostic", "content": f"report:{query}"}
        yield {"type": "done", "agent": "diagnostic"}


def test_non_handoff_query_keeps_conversation_stream():
    orch = Orchestrator(conversation_agent=_EchoConversationAgent(), diagnostic_agent=_NoGoAgent())
    events = list(orch.execute("今天天气如何", history=[], mode="conversation"))
    assert [(e["type"], e.get("content", "")) for e in events] == [
        ("message", "echo:今天天气如何"),
        ("done", ""),
    ]


def test_non_handoff_query_keeps_diagnostic_stream():
    orch = Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_EchoDiagnosticAgent())
    events = list(orch.execute("扫地机异响", mode="diagnostic"))
    assert [(e["type"], e.get("content", "")) for e in events] == [
        ("report", "report:扫地机异响"),
        ("done", ""),
    ]


# ===================================================================== SSE 桥不丢关键事件
def _consume_after_warmup(stream, warmup_seconds: float = 0.5) -> list[dict]:
    async def consume():
        ait = stream.__aiter__()
        out = [await ait.__anext__()]
        await asyncio.sleep(warmup_seconds)
        async for item in ait:
            out.append(item)
        return out

    return asyncio.run(asyncio.wait_for(consume(), timeout=5))


def test_handoff_event_survives_full_queue():
    """业务队列被普通事件填满并发生丢弃后，handoff_suggested 仍恰好送达一次且为最后一帧。"""
    total = 60

    def runner():
        for i in range(total):
            yield {"type": "message", "content": str(i)}
        yield make_event(
            "handoff_suggested",
            agent="orchestrator",
            content="已收到您的转人工请求",
            reason="user_request",
        )

    from api.streaming import build_sse_stream

    stream = build_sse_stream(
        runner, request_id="rid-handoff-full", queue_maxsize=3, heartbeat_seconds=30.0,
    )
    out = _consume_after_warmup(stream)

    handoff_frames = [
        item for item in out
        if item.get("event") == "message" and '"handoff_suggested"' in item.get("data", "")
    ]
    assert len(handoff_frames) == 1, (
        f"队列满时 handoff 事件绝不能丢（收到 {len(handoff_frames)} 个）"
    )
    payload = json.loads(handoff_frames[0]["data"])
    assert payload["data"]["reason"] == "user_request"

    # 业务背压确实发生过：收到的普通事件远少于生产数
    normal = [
        item for item in out
        if item.get("event") == "message" and '"handoff' not in item.get("data", "")
    ]
    assert 0 < len(normal) < total, f"普通事件应存在丢弃背压（收到 {len(normal)}/{total}）"
    # handoff 是流结束前的最后一个事件帧
    assert out[-1] is handoff_frames[0] or out[-1] == handoff_frames[0]


# ===================================================================== API 层透传
class _HandoffOnlyOrchestrator:
    def execute(self, query, history=None, mode=None, conversation_id=None):
        yield make_event(
            "handoff_suggested",
            agent="orchestrator",
            content="已收到您的转人工请求，本次会话已标记为需要人工跟进。",
            reason="user_request",
            keyword="转人工",
        )
        yield make_event("done", agent="orchestrator")


@pytest.fixture
def handoff_chat_client(api_client):
    """注入只产 handoff 事件的桩 orchestrator（api_client 的 monkeypatch 负责还原）。"""
    from api.main import app

    app.state.container._orchestrator = _HandoffOnlyOrchestrator()
    return api_client


def test_chat_sse_passes_through_handoff_event(handoff_chat_client):
    resp = handoff_chat_client.post("/api/chat", json={"query": "转人工"})
    assert resp.status_code == 200
    assert '"handoff_suggested"' in resp.text
    assert '"done"' in resp.text


def test_chat_sync_returns_need_handoff_true(handoff_chat_client):
    resp = handoff_chat_client.post("/api/chat/sync", json={"query": "转人工"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["need_handoff"] is True
    assert body["answer"] == "", "handoff 提示不应混入 answer"


def test_chat_sync_normal_flow_need_handoff_false(api_client):
    from tests.conftest import CannedOrchestrator

    from api.main import app

    app.state.container._orchestrator = CannedOrchestrator()
    resp = api_client.post("/api/chat/sync", json={"query": "你好"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["need_handoff"] is False
    assert body["answer"] == "这是测试回复。"


# ===================================================================== 转人工闭环（Orchestrator 级）
class _RecordingTicketRepo:
    """记录调用次数的假工单仓库：返回固定工单；fail=True 时模拟存储故障。"""

    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("sqlite disk I/O error")
        return ({
            "ticket_id": "ticket-1",
            "conversation_id": kwargs.get("conversation_id") or "conv-x",
            "user_question": kwargs.get("user_question", ""),
            "status": "pending",
            "created_at": "2026-08-27T00:00:00+00:00",
            "updated_at": "2026-08-27T00:00:00+00:00",
        }, "secret-access-key-123456789012345678901")

    # 人工接管期间活动工单检测：默认无活动工单（普通查询保持模型介入）
    def get_active_by_conversation(self, conversation_id):
        self.activity_checks = getattr(self, "activity_checks", 0) + 1
        return None

    def append_user_message(self, conversation_id, content):
        return None


@pytest.mark.parametrize("query,keyword", [
    ("帮我转人工", "转人工"),
    ("找人工客服", "人工客服"),
    ("我要联系人工", "联系人工"),
    ("请转接人工", "转接人工"),
    ("我要人工", "我要人工"),
])
def test_each_keyword_creates_ticket_and_emits_handoff_created(query, keyword):
    """每个关键词都命中 → 单轮创建一张工单 → 事件流含 handoff_created。"""
    repo = _RecordingTicketRepo()
    orch = Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
                        handoff_tickets=repo)
    events = list(orch.execute(query))
    assert [e["type"] for e in events] == ["handoff_suggested", "handoff_created", "done"]
    assert len(repo.calls) == 1, f"{query} 单轮应只创建一次工单"
    assert repo.calls[0]["handoff_reason"] == "user_request"

    created = events[1]
    data = created.get("data", {})
    assert data["ticket_id"] == "ticket-1"
    assert data["status"] == "pending"
    assert data["message"], "事件 message 应可对用户展示"
    # 明文 access_key 仅在 handoff_created 事件下发一次，且与 ticket_id 匹配
    assert data["access_key"] == "secret-access-key-123456789012345678901"
    assert data["ticket_id"] == "ticket-1"
    # 其余事件（suggested / done）以及哈希不得出现 access_key
    for other in (events[0], events[2]):
        assert "access_key" not in json.dumps(other)
    assert "access_key_hash" not in json.dumps(events)


def test_handoff_saves_conversation_context():
    """工单保存最近对话、摘要、原始问题、reason；事件与仓库调用保持一致。"""
    repo = _RecordingTicketRepo()
    orch = Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
                        handoff_tickets=repo)
    history = [
        {"role": "user", "content": "机器不工作了"},
        {"role": "assistant", "content": "请提供更多信息"},
    ]
    list(orch.execute("我要人工", history=history))
    assert len(repo.calls) == 1
    kwargs = repo.calls[0]
    assert kwargs["user_question"] == "我要人工"
    assert kwargs["conversation_id"], "应从 history 推导会话标识"
    assert kwargs["recent_conversations"] == history
    assert "机器不工作了" in kwargs["conversation_summary"]
    assert "请提供更多信息" in kwargs["conversation_summary"]
    assert kwargs["handoff_reason"] == "user_request"
    assert kwargs["retrieval_sources"] is None
    assert kwargs["diagnostic_result"] == ""


def test_handoff_create_failure_emits_unified_error():
    """存储异常 → 统一 error 事件（固定 error_code，不泄漏内部细节）。"""
    repo = _RecordingTicketRepo(fail=True)
    orch = Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
                        handoff_tickets=repo)
    events = list(orch.execute("转人工"))
    assert [e["type"] for e in events] == ["handoff_suggested", "error", "done"]
    assert events[1]["data"]["error_code"] == "HANDOFF_TICKET_CREATE_FAILED"
    assert events[1]["content"], "错误事件应携带对用户可见的安全提示"
    assert "access_key" not in json.dumps(events)


def test_handoff_single_round_creates_ticket_once():
    """同一轮（单次 execute）不得重复创建工单：每次 execute 恰好一次 create。"""
    repo = _RecordingTicketRepo()
    orch = Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
                        handoff_tickets=repo)
    first = list(orch.execute("我要人工"))
    second = list(orch.execute("我要人工"))
    assert [e["type"] for e in first] == ["handoff_suggested", "handoff_created", "done"]
    assert [e["type"] for e in second] == ["handoff_suggested", "handoff_created", "done"]
    assert len(repo.calls) == 2, "每轮请求各建一张，同一轮内不重复"


def test_normal_query_does_not_touch_ticket_repo():
    """普通问题不触发转人工：仓库零调用，事件流不含 handoff。"""
    repo = _RecordingTicketRepo()
    orch = Orchestrator(conversation_agent=_EchoConversationAgent(), diagnostic_agent=_NoGoAgent(),
                        handoff_tickets=repo)
    events = list(orch.execute("今天天气如何", history=[], mode="conversation"))
    assert all(e["type"] not in ("handoff_suggested", "handoff_created") for e in events)
    assert repo.calls == []


def _real_handoff_orchestrator() -> Orchestrator:
    from agent.handoff_tickets import HandoffTicketService

    return Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
                        handoff_tickets=HandoffTicketService())


def test_chat_sse_real_handoff_emits_created_with_access_key(api_client, tmp_path, monkeypatch):
    """真实 HandoffTicketService + SQLite：SSE 流产出 handoff_created，且明文 access_key 只出现在该事件帧、未落入日志/哈希。"""
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "sse.db"))
    import logging
    from api.main import app

    records: list[str] = []
    original = logging.getLogger("agent.orchestrator")
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    original.addHandler(handler)
    try:
        app.state.container._orchestrator = _real_handoff_orchestrator()
        resp = api_client.post("/api/chat", json={"query": "请转接人工"})
    finally:
        original.removeHandler(handler)

    assert resp.status_code == 200
    assert '"handoff_suggested"' in resp.text
    assert '"handoff_created"' in resp.text
    assert '"ticket_id"' in resp.text
    assert '"access_key"' in resp.text, "自动建单事件应包含一次性明文 access_key"
    assert '"pending"' in resp.text
    assert "access_key_hash" not in resp.text, "hash 不得出现在 SSE"
    # access_key 不得出现在 orchestor 日志中
    assert not any("secret-access-key" in line or "access_key" in line for line in records), (
        "access_key 不得写入日志"
    )


def test_chat_sync_real_handoff_returns_need_handoff_and_persists(api_client, tmp_path, monkeypatch):
    """真实闭环：/chat/sync 返回 need_handoff=True，工单真正落库。"""
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "sync.db"))
    from api.handoff_store import HandoffStore
    from api.main import app

    app.state.container._orchestrator = _real_handoff_orchestrator()
    resp = api_client.post("/api/chat/sync", json={"query": "我要人工"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["need_handoff"] is True

    _, items = HandoffStore().list_tickets()
    assert len(items) == 1
    assert items[0]["user_question"] == "我要人工"
    assert items[0]["status"] == "pending"
    assert "access_key" not in items[0]


def test_auto_create_event_access_key_matches_ticket(tmp_path, monkeypatch):
    """自动建单：事件中的 access_key 可凭其取回同一 ticket_id 的工单。"""
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "match.db"))
    from agent.handoff_tickets import HandoffTicketService
    from api.handoff_store import HandoffStore

    orch = Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
                        handoff_tickets=HandoffTicketService())
    events = list(orch.execute("我要人工"))
    created = events[1]
    assert created["type"] == "handoff_created"
    data = created["data"]

    ticket = HandoffStore().get_by_access_key(data["ticket_id"], data["access_key"])
    assert ticket is not None, "事件下发的 access_key 应能取回该工单"
    assert ticket["ticket_id"] == data["ticket_id"]
    assert "access_key" not in ticket, "工单记录内不得含明文 access_key"
    assert "access_key_hash" not in ticket


def test_auto_create_access_key_unique_per_ticket(tmp_path, monkeypatch):
    """每次自动建单生成的 access_key 唯一且为高熵随机串。"""
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "uniq.db"))
    from agent.handoff_tickets import HandoffTicketService

    orch = Orchestrator(conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
                        handoff_tickets=HandoffTicketService())
    first = list(orch.execute("我要人工"))[1]["data"]
    second = list(orch.execute("人工客服"))[1]["data"]
    assert first["access_key"] and second["access_key"]
    assert first["access_key"] != second["access_key"], "每次自动建单凭证必须唯一"
    assert len(first["access_key"]) == 43 and len(second["access_key"]) == 43


def test_chat_sync_handoff_create_failure_uniform_error(api_client, monkeypatch):
    """sync 路径存储故障 → 统一 500 错误结构。"""
    from api.main import app

    app.state.container._orchestrator = Orchestrator(
        conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
        handoff_tickets=_RecordingTicketRepo(fail=True),
    )
    resp = api_client.post("/api/chat/sync", json={"query": "转人工"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == "HANDOFF_TICKET_CREATE_FAILED"
    assert body["safe_message"]
    assert body["request_id"]


# ===================================================================== 存储层
@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "handoffs.db"))
    from api.handoff_store import HandoffStore

    return HandoffStore()


def test_store_requires_env_configured_path(monkeypatch):
    monkeypatch.delenv("HANDOFF_DB_PATH", raising=False)
    from api.handoff_store import HandoffStore, HandoffStoreError

    with pytest.raises(HandoffStoreError):
        HandoffStore()


def test_store_create_and_get_roundtrip(store):
    sources = [{"document": "故障排除.txt", "chunk_id": 3, "score": 0.42}]
    ticket, access_key = store.create(
        user_question="扫地机无法回充",
        conversation_id="conv-1",
        retrieval_sources=sources,
        diagnostic_result="中sensor 异常",
        handoff_reason="user_request",
    )
    assert ticket["status"] == "pending"
    assert ticket["human_reply"] is None
    got = store.get(ticket["ticket_id"])
    assert got == ticket
    assert got["retrieval_sources"] == sources, "sources 应序列化入库后还原"


def test_store_creates_live_messages_and_reply_appends_human(store):
    """建单初始化实时对话为 user；回复追加 human 消息。"""
    ticket, _ = store.create(user_question="有没有优惠券", conversation_id="conv-live")
    assert [(m["role"], m["content"]) for m in ticket["live_messages"]] == [
        ("user", "有没有优惠券")
    ]
    updated = store.reply(ticket["ticket_id"], "有的，新用户首单9折", status="processing")
    assert [(m["role"], m["content"]) for m in updated["live_messages"]] == [
        ("user", "有没有优惠券"),
        ("human", "有的，新用户首单9折"),
    ]


def test_store_active_by_conversation_and_append_user_message(store):
    """活动工单按会话命中；追加用户消息写实时日志；结束后不再命中（模型可复出）。"""
    t, _ = store.create(user_question="问题A", conversation_id="conv-active")
    assert store.get_active_by_conversation("conv-active") is not None
    store.append_user_message("conv-active", "问题B: 有没有优惠券")
    got = store.get(t["ticket_id"])
    assert [m["content"] for m in got["live_messages"]] == ["问题A", "问题B: 有没有优惠券"]
    store.reply(t["ticket_id"], "处理完毕", status="resolved")
    assert store.get_active_by_conversation("conv-active") is None, "已解决后模型应重新介入"


def test_active_handoff_suppresses_model_and_appends_user_message():
    """人工接管期间：模型停手（NoGo 哨兵不抛即未调用），用户新消息入实时日志，并产出手动接管事件。"""

    class Repo:
        def __init__(self):
            self.appended: list[tuple[str, str]] = []

        def get_active_by_conversation(self, cid):
            if cid == "conv-active":
                return {"ticket_id": "t1", "status": "processing"}
            return None

        def append_user_message(self, cid, content):
            self.appended.append((cid, content))
            return {}

    repo = Repo()
    orch = Orchestrator(
        conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(), handoff_tickets=repo
    )
    void_conversation = Orch = None  # noqa: F841
    events = list(orch.execute("有没有优惠券", history=[{"role": "user", "content": "conv-active"}]))
    assert repo.appended == [("conv-active", "有没有优惠券")]
    assert [e["type"] for e in events] == ["handoff_human_service", "done"]
    assert "人工" in events[0].get("content", "")


def test_store_access_key_high_entropy_and_unique(store):
    t1, k1 = store.create(user_question="q1")
    t2, k2 = store.create(user_question="q2")
    assert k1 != k2, "每次创建应生成独立随机 access_key"
    for key in (k1, k2):
        # token_urlsafe(32) → 43 字符 URL 安全文本（约 192 bit 熵）
        assert len(key) == 43
        assert all(c.isalnum() or c in "-_" for c in key)


def test_store_db_persists_only_hash(store, tmp_path):
    import hashlib
    import sqlite3

    ticket, access_key = store.create(user_question="q", conversation_id="conv-h")
    db_path = tmp_path / "handoffs.db"

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT access_key_hash FROM handoff_tickets WHERE ticket_id = ?",
            (ticket["ticket_id"],),
        ).fetchone()
    expected = hashlib.sha256(access_key.encode("utf-8")).hexdigest()
    assert row[0] == expected, "库中应只存 SHA-256 哈希"
    raw = db_path.read_bytes()
    assert access_key.encode("utf-8") not in raw, "明文 access_key 不得出现在数据库文件中"


def test_store_get_missing_returns_none(store):
    assert store.get("no-such-ticket") is None


def test_store_reply_updates_reply_and_status(store):
    t, _ = store.create(user_question="q")
    updated = store.reply(t["ticket_id"], human_reply="已安排客服跟进")
    assert updated["status"] == "resolved"
    assert updated["human_reply"] == "已安排客服跟进"
    assert updated["updated_at"] >= updated["created_at"]

    custom, _ = store.create(user_question="q2")
    updated2 = store.reply(custom["ticket_id"], "处理中", status="processing")
    assert updated2["status"] == "processing"

    assert store.reply("missing", "hi") is None


def test_store_list_filters_and_paginates(store):
    a, _ = store.create(user_question="a")
    b, _ = store.create(user_question="b")
    store.create(user_question="c")
    store.reply(b["ticket_id"], "ok")

    total, items = store.list_tickets(limit=10)
    assert total == 3 and len(items) == 3

    total_pending, pending = store.list_tickets(status="pending")
    assert total_pending == 2
    assert {t["ticket_id"] for t in pending} >= {a["ticket_id"]}

    page_total, page = store.list_tickets(limit=1, offset=1)
    assert page_total == 3 and len(page) == 1


def test_store_get_by_access_key(store):
    ticket, access_key = store.create(user_question="q", conversation_id="cid")

    assert store.get_by_access_key(ticket["ticket_id"], access_key) is not None
    # 错误 key / 空 key / 不存在工单 / conversation_id 变化不影响凭证校验
    assert store.get_by_access_key(ticket["ticket_id"], "wrong-key") is None
    assert store.get_by_access_key(ticket["ticket_id"], "") is None
    assert store.get_by_access_key("missing-ticket", access_key) is None


# ===================================================================== 接口层
ADMIN_TOKEN = "adm-secret-123"


def _login_session(api_client) -> str:
    """登录并返回会话 Cookie 值。

    返回的 Cookie 值用于手动携带到管理接口：生产 Cookie 带 Secure 标记，
    httpx 的 Cookie jar 不会在 http 下回传 Secure Cookie，故测试显式传入。
    """
    resp = api_client.post("/api/admin/login", json={"admin_token": ADMIN_TOKEN})
    assert resp.status_code == 200, resp.text
    set_cookie = resp.headers.get("set-cookie", "")
    assert set_cookie, "登录成功应下发会话 Cookie"
    return set_cookie.split(";", 1)[0].split("=", 1)[1]


def _session_headers(sid: str) -> dict:
    return {"Cookie": f"admin_session={sid}"}


@pytest.fixture
def tickets_api_client(api_client, tmp_path, monkeypatch):
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "tickets.db"))
    monkeypatch.setenv("ADMIN_TOKEN", "adm-secret-123")
    return api_client


def test_create_and_user_get_own_ticket(tickets_api_client):
    resp = tickets_api_client.post("/api/handoff", json={
        "user_question": "机器一直报错 E03",
        "conversation_id": "conv-a",
        "conversation_summary": "用户反馈故障码",
        "retrieval_sources": [{"document": "故障排除.txt", "score": 0.31}],
        "diagnostic_result": "疑似门锁传感器故障",
        "handoff_reason": "rag_low_confidence",
    })
    assert resp.status_code == 201
    created = resp.json()
    assert created["status"] == "pending"
    assert created["conversation_id"] == "conv-a"
    assert created["ticket_id"]
    # 创建响应必须返回明文 access_key，且不得出现哈希
    access_key = created.get("access_key", "")
    assert len(access_key) == 43, "创建响应应一次性返回明文 access_key"
    assert "access_key_hash" not in created

    got = tickets_api_client.get(
        f"/api/handoff/{created['ticket_id']}", params={"access_key": access_key}
    )
    assert got.status_code == 200
    body = got.json()
    assert body["human_reply"] is None
    assert body["retrieval_sources"][0]["document"] == "故障排除.txt"
    # 后续查询不再返回凭证材料（明文或哈希）
    assert "access_key" not in body
    assert "access_key_hash" not in body


def test_conversation_id_is_not_a_credential(tickets_api_client):
    """conversation_id 仅用于关联会话：即使携带它也查不到工单，正确 key 才可以。"""
    created = tickets_api_client.post(
        "/api/handoff", json={"user_question": "q", "conversation_id": "mine"}
    ).json()

    with_cid_only = tickets_api_client.get(
        f"/api/handoff/{created['ticket_id']}", params={"conversation_id": "mine"}
    )
    wrong_cid_right_key = tickets_api_client.get(
        f"/api/handoff/{created['ticket_id']}",
        params={"conversation_id": "something-else", "access_key": created["access_key"]},
    )
    assert with_cid_only.status_code == 404, "仅凭 conversation_id 不能访问工单"
    assert wrong_cid_right_key.status_code == 200, "conversation_id 不参与鉴权"


def test_user_auth_failures_uniform_404(tickets_api_client):
    """错误 key / 缺失 key / 空 key / 不存在工单 → 统一 404 与统一错误体。"""
    created = tickets_api_client.post(
        "/api/handoff", json={"user_question": "q"}
    ).json()
    tid = created["ticket_id"]

    cases = [
        ("wrong key", (f"/api/handoff/{tid}", {"params": {"access_key": "totally-wrong"}})),
        ("missing key", (f"/api/handoff/{tid}", {})),
        ("empty key", (f"/api/handoff/{tid}", {"params": {"access_key": ""}})),
        (
            "unknown ticket with plausible key",
            ("/api/handff/unknown".replace("handff", "handoff"), {
                "params": {"access_key": "a" * 43}
            }),
        ),
    ]
    for label, (url, kwargs) in cases:
        resp = tickets_api_client.get(url, **kwargs)
        assert resp.status_code == 404, f"{label} 应返回 404"
        body = resp.json()
        assert body["error_code"] == "HANDOFF_TICKET_NOT_FOUND"
        assert body["safe_message"] == "工单不存在。"
        assert body["request_id"], f"{label} 响应应携带 request_id"


def test_admin_requires_session(tickets_api_client):
    no_cookie = tickets_api_client.get("/api/admin/handoffs")
    bad_cookie = tickets_api_client.get("/api/admin/handoffs", headers={"Cookie": "admin_session=invalid"})
    sid = _login_session(tickets_api_client)
    good = tickets_api_client.get("/api/admin/handoffs", headers=_session_headers(sid))

    assert no_cookie.status_code == 401
    assert no_cookie.json()["error_code"] == "HANDOFF_ADMIN_REQUIRED"
    assert bad_cookie.status_code == 401
    assert good.status_code == 200


def test_admin_login_success_sets_secure_http_only_cookie(tickets_api_client):
    resp = tickets_api_client.post("/api/admin/login", json={"admin_token": ADMIN_TOKEN})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "admin_token" not in body, "登录响应不得回显令牌"
    set_cookie = resp.headers["set-cookie"]
    assert "admin_session=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=lax" in set_cookie


def test_admin_login_rejects_wrong_token(tickets_api_client):
    resp = tickets_api_client.post("/api/admin/login", json={"admin_token": "wrong-token"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "HANDOFF_ADMIN_REQUIRED"
    assert "set-cookie" not in resp.headers


def test_admin_session_expired_requires_relogin(tickets_api_client):
    sid = _login_session(tickets_api_client)
    assert tickets_api_client.get("/api/admin/handoffs", headers=_session_headers(sid)).status_code == 200

    from api.admin_session import session_store

    session_store.force_expire(sid)
    assert tickets_api_client.get("/api/admin/handoffs", headers=_session_headers(sid)).status_code == 401


def test_admin_login_disabled_when_token_unset(api_client, tmp_path, monkeypatch):
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "tickets.db"))
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    resp = api_client.post("/api/admin/login", json={"admin_token": "x"})
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "HANDOFF_ADMIN_DISABLED"


def test_admin_disabled_when_token_unset(api_client, tmp_path, monkeypatch):
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "tickets.db"))
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)

    resp = api_client.get("/api/admin/handoffs")
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "HANDOFF_ADMIN_DISABLED"


def test_store_unavailable_returns_503(api_client, monkeypatch):
    monkeypatch.delenv("HANDOFF_DB_PATH", raising=False)
    resp = api_client.post("/api/handoff", json={"user_question": "q"})
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "HANDOFF_DB_NOT_CONFIGURED"


def test_admin_list_detail_and_reply_flow(tickets_api_client):
    sid = _login_session(tickets_api_client)
    t1 = tickets_api_client.post("/api/handoff", json={
        "user_question": "问题一", "conversation_id": "c1",
    }).json()
    t2 = tickets_api_client.post("/api/handoff", json={
        "user_question": "问题二", "conversation_id": "c2",
    }).json()
    t3 = tickets_api_client.post("/api/handoff", json={
        "user_question": "问题三", "conversation_id": "c3",
    }).json()

    listed = tickets_api_client.get("/api/admin/handoffs", headers=_session_headers(sid)).json()
    assert listed["total"] == 3 and len(listed["items"]) == 3
    # 管理接口也不得回显凭证材料（明文或哈希）
    assert all("access_key" not in item and "access_key_hash" not in item
               for item in listed["items"])

    filtered = tickets_api_client.get(
        "/api/admin/handoffs", params={"status": "pending"}, headers=_session_headers(sid)
    ).json()
    assert filtered["total"] == 3

    paged = tickets_api_client.get(
        "/api/admin/handoffs", params={"limit": 2, "offset": 1}, headers=_session_headers(sid)
    ).json()
    assert paged["total"] == 3 and len(paged["items"]) == 2

    bad_status = tickets_api_client.get(
        "/api/admin/handoffs", params={"status": "bogus"}, headers=_session_headers(sid)
    )
    assert bad_status.status_code == 422

    detail = tickets_api_client.get(f"/api/admin/handoffs/{t1['ticket_id']}", headers=_session_headers(sid))
    assert detail.status_code == 200 and detail.json()["user_question"] == "问题一"
    assert "access_key" not in detail.json() and "access_key_hash" not in detail.json()

    admin_missing = tickets_api_client.get("/api/admin/handoffs/nope", headers=_session_headers(sid))
    assert admin_missing.status_code == 404

    replied = tickets_api_client.post(
        f"/api/admin/handoffs/{t2['ticket_id']}/reply",
        headers=_session_headers(sid),
        json={"human_reply": "已联系售后上门"},
    )
    assert replied.status_code == 200
    body = replied.json()
    assert body["status"] == "resolved"
    assert body["human_reply"] == "已联系售后上门"

    explicit = tickets_api_client.post(
        f"/api/admin/handoffs/{t3['ticket_id']}/reply",
        headers=_session_headers(sid),
        json={"human_reply": "处理中", "status": "processing"},
    )
    assert explicit.json()["status"] == "processing"

    reply_missing = tickets_api_client.post(
        "/api/admin/handoffs/nope/reply", headers=_session_headers(sid), json={"human_reply": "x"}
    )
    assert reply_missing.status_code == 404

    invalid_body = tickets_api_client.post(
        f"/api/admin/handoffs/{t2['ticket_id']}/reply",
        headers=_session_headers(sid),
        json={"human_reply": "x", "status": "wat"},
    )
    assert invalid_body.status_code == 422


def test_create_rejects_blank_question(tickets_api_client):
    resp = tickets_api_client.post("/api/handoff", json={"user_question": ""})
    assert resp.status_code == 422


def _sse_payloads(text: str) -> list[dict]:
    """解析 SSE 文本，返回各 `data:` 帧的 JSON 载荷（.type/.data 结构）。"""
    out: list[dict] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.startswith("data:"):
            current.append(line[5:].strip())
        elif current:
            try:
                out.append(json.loads("".join(current)))
            except (ValueError, TypeError):
                pass
            current = []
    if current:
        try:
            out.append(json.loads("".join(current)))
        except (ValueError, TypeError):
            pass
    return out


def test_full_roundtrip_resolved_visible_to_user(api_client, tmp_path, monkeypatch):
    """端到端闭环：用户转人工 → SSE 自动建单（含凭证）→ 管理员登录/列表/详情/回复 → 用户凭 access_key 看到 resolved。

    同时覆盖：SSE 自动建单凭证可查询、页面与日志不暴露 access_key、
    普通用户无法直接访问管理接口、回复后工单状态变为 resolved。
    """
    monkeypatch.setenv("HANDOFF_DB_PATH", str(tmp_path / "roundtrip.db"))
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    from agent.handoff_tickets import HandoffTicketService
    from agent.orchestrator import Orchestrator
    from api.main import app

    app.state.container._orchestrator = Orchestrator(
        conversation_agent=_NoGoAgent(), diagnostic_agent=_NoGoAgent(),
        handoff_tickets=HandoffTicketService(),
    )

    # 1) 普通用户触达转人工：SSE 自动建单，事件须含 ticket_id/access_key/status
    resp = api_client.post("/api/chat", json={"query": "我要人工"})
    assert resp.status_code == 200
    created = next(r for r in _sse_payloads(resp.text) if r.get("type") == "handoff_created")
    data = created["data"]
    ticket_id = data["ticket_id"]
    access_key = data["access_key"]
    assert data["status"] == "pending"

    # 2) 普通用户无法访问管理接口（无会话 Cookie → 401）
    assert api_client.get("/api/admin/handoffs").status_code == 401

    # 3) 管理员登录并建立会话
    sid = _login_session(api_client)

    # 4) 管理列表包含该工单
    listed = api_client.get("/api/admin/handoffs", headers=_session_headers(sid)).json()
    assert any(t["ticket_id"] == ticket_id for t in listed["items"])

    # 5) 工单详情与上下文，管理侧不回显凭证
    detail = api_client.get(f"/api/admin/handoffs/{ticket_id}", headers=_session_headers(sid)).json()
    assert detail["user_question"] == "我要人工"
    assert "access_key" not in detail and "access_key_hash" not in detail

    # 6) 管理员输入人工回复，状态置为 resolved
    replied = api_client.post(
        f"/api/admin/handoffs/{ticket_id}/reply",
        headers=_session_headers(sid),
        json={"human_reply": "已为您转接售后，问题已解决。", "status": "resolved"},
    )
    assert replied.status_code == 200
    assert replied.json()["status"] == "resolved"

    # 7) 用户凭一次性的 access_key 再查询：状态 resolved + 人工回复可见，密码材料不再回显
    got = api_client.get(f"/api/handoff/{ticket_id}", params={"access_key": access_key})
    assert got.status_code == 200
    body = got.json()
    assert body["status"] == "resolved"
    assert body["human_reply"] == "已为您转接售后，问题已解决。"
    assert "access_key" not in body and "access_key_hash" not in body
