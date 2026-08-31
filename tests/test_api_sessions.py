"""API 会话集成测试：DiagnoseRequest / ChatRequest 的 user_id 与 conversation_id。

覆盖验收点：
- 诊断请求可以接收稳定的 user_id（未传时默认演示用户 1001）
- DiagnoseRequest 支持 conversation_id
- 同一会话后续请求能读取历史（SSE 首个事件携带 conversation_id）
- 不同 user_id 读取他人会话 → 403 + CONVERSATION_ACCESS_DENIED
- 旧请求格式 {"query": "..."} 仍可用且自动新建会话
"""
import contextlib
import json

import pytest


class _HistoryEchoOrchestrator:
    """记录收到的 history / conversation_id，并把历史回显到 message 事件。"""

    def __init__(self):
        self.seen: list[dict] = []
        self.conversation_ids: list[str] = []

    def execute(self, query, history=None, mode=None, conversation_id=None):
        self.seen.append(list(history or []))
        self.conversation_ids.append(conversation_id)
        history_hint = "｜".join(m.get("content", "") for m in (history or []))

        def gen():
            yield {"type": "message", "agent": "conversation",
                   "content": f"回复：{query}；历史[{history_hint}]"}
            yield {"type": "done", "agent": "conversation", "content": ""}
        return gen()


@pytest.fixture
def history_client(api_client, monkeypatch):
    """注入会回显历史的桩 orchestrator，返回 (client, orchestrator)。"""
    from api.main import app

    orchestrator = _HistoryEchoOrchestrator()
    app.state.container._orchestrator = orchestrator
    return api_client, orchestrator


def _sse_events(resp):
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data:"):
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line[5:].strip()))
    return events


# ----------------------------------------------------------------- /api/chat/sync
def test_chat_sync_creates_conversation_and_returns_id(history_client):
    client, orchestrator = history_client

    resp = client.post("/api/chat/sync", json={"query": "机器清洁效率很低"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"].startswith("conv-")
    assert body["user_id"] == "1001", "未传 user_id 时使用默认演示用户"
    assert "机器清洁效率很低" in body["answer"]


def test_chat_sync_second_request_reads_history(history_client):
    client, orchestrator = history_client

    first = client.post("/api/chat/sync", json={"query": "机器清洁效率很低"}).json()
    cid = first["conversation_id"]
    second = client.post("/api/chat/sync", json={
        "query": "还出现过 E03",
        "user_id": "1001",
        "conversation_id": cid,
    }).json()

    assert second["conversation_id"] == cid, "同一会话复用同一 ID"
    assert "机器清洁效率很低" in second["answer"], "第二轮应读到第一轮历史"
    assert orchestrator.seen[1], "第二轮请求携带了历史"


def test_chat_sync_other_user_denied(history_client):
    client, _ = history_client
    cid = client.post("/api/chat/sync", json={"query": "我的问题"}).json()["conversation_id"]

    denied = client.post("/api/chat/sync", json={
        "query": "别人的问题", "user_id": "1002", "conversation_id": cid,
    })
    assert denied.status_code == 403
    body = denied.json()
    assert body["error_code"] == "CONVERSATION_ACCESS_DENIED"
    assert "无权访问" in body["safe_message"]


def test_chat_sync_legacy_request_format_still_works(history_client):
    """旧格式 {"query": "..."} 仍可用：自动使用默认演示用户 + 新建会话。"""
    client, orchestrator = history_client
    resp = client.post("/api/chat/sync", json={"query": "你好"})
    assert resp.status_code == 200
    assert resp.json()["conversation_id"]
    assert orchestrator.seen[0] == [], "无 conversation_id 时沿用客户端 history（此处为空）"


def test_chat_sync_legacy_history_is_passed_through(history_client):
    """未携带 conversation_id 时，客户端传入的 history 仍透传给 Agent（旧行为不变）。"""
    client, orchestrator = history_client
    history = [{"role": "user", "content": "上次的滤网"}, {"role": "assistant", "content": "用软布擦拭"}]
    resp = client.post("/api/chat/sync", json={"query": "那多久换一次", "history": history})
    assert resp.status_code == 200
    assert orchestrator.seen[0] == history


# ----------------------------------------------------------------- /api/diagnose
def test_diagnose_accepts_user_id_and_conversation_id(api_client, monkeypatch):
    from api.main import app

    captured = {}

    class TrackingOrchestrator:
        def execute(self, query, history=None, mode=None, conversation_id=None):
            captured["history"] = list(history or [])
            captured["conversation_id"] = conversation_id

            def gen():
                yield {"type": "plan", "agent": "diagnostic", "content": "", "data": {"steps": ["查状态"]}}
                yield {"type": "report", "agent": "diagnostic", "content": "## 诊断报告\n设备ID：device-1002"}
                yield {"type": "done", "agent": "diagnostic", "content": ""}
            return gen()

    app.state.container._orchestrator = TrackingOrchestrator()

    resp = api_client.post("/api/diagnose", json={
        "query": "清洁效率很低", "user_id": "1002", "conversation_id": None,
    })
    assert resp.status_code == 200
    events = _sse_events(resp)
    session_events = [e for e in events if e["type"] == "session"]
    assert session_events, "SSE 首个事件应为 session 事件"
    cid = session_events[0]["data"]["conversation_id"]
    assert session_events[0]["data"]["user_id"] == "1002"
    assert captured["conversation_id"] == cid

    # 第二轮：同一会话应读到第一轮的诊断历史（故障描述 + 报告）
    second = api_client.post("/api/diagnose", json={
        "query": "还出现过 E03", "user_id": "1002", "conversation_id": cid,
    })
    assert second.status_code == 200
    assert any("清洁效率很低" in str(m) for m in captured["history"]), "第二轮应读到第一轮历史"


def test_diagnose_other_user_denied(api_client):
    first = api_client.post("/api/diagnose", json={"query": "设备无法启动"})
    cid = [e for e in _sse_events(first) if e["type"] == "session"][0]["data"]["conversation_id"]

    denied = api_client.post("/api/diagnose", json={
        "query": "设备无法启动", "user_id": "1003", "conversation_id": cid,
    })
    assert denied.status_code == 403
    assert denied.json()["error_code"] == "CONVERSATION_ACCESS_DENIED"


def test_chat_sse_first_event_carries_conversation_id(history_client):
    client, _ = history_client
    resp = client.post("/api/chat", json={"query": "你好"})
    assert resp.status_code == 200
    events = _sse_events(resp)
    assert events and events[0]["type"] == "session"
    assert events[0]["data"]["conversation_id"].startswith("conv-")
    assert events[-1]["type"] == "done"
