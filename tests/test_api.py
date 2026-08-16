"""API 接口测试：health / chat(SSE) / diagnose(SSE) / knowledge upload / knowledge rebuild。

对话与诊断接口在路由层把 get_orchestrator 替换为固定事件流（见 conftest.api_client），
避免构造需要真实模型的 ConversationAgent。上传/重建接口额外打桩以隔离文件系统与向量库。
"""
import contextlib
import json
import os


def test_health(api_client):
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "reranker_enabled" in body


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
        def __init__(self, *a, **k):
            pass

        def load_document(self):
            return None

        @property
        def vector_store(self):
            class _Coll:
                def count(self):
                    return 12

            class _VS:
                _collection = _Coll()

            return _VS()

    monkeypatch.setattr("api.routes.knowledge.VectorStoreService", lambda *a, **k: _StubVS())
    resp = api_client.post("/api/knowledge/rebuild")
    assert resp.status_code == 200
    assert resp.json()["chunk_count"] == 12


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


def test_sse_error_event_state_machine(api_client, monkeypatch):
    """验证生成器异常时产生 error 事件"""
    def broken_orchestrator():
        class _Orch:
            def execute(self, *a, **kw):
                def gen():
                    yield {"type": "message", "agent": "test", "content": "开始"}
                    raise RuntimeError("模拟异常")
                return gen()
        return _Orch()

    monkeypatch.setattr("api.routes.conversation.get_orchestrator", broken_orchestrator)
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
