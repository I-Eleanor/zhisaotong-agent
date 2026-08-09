"""API 接口测试：health / chat(SSE) / diagnose(SSE) / knowledge upload / knowledge rebuild。

对话与诊断接口在路由层把 get_orchestrator 替换为固定事件流（见 conftest.api_client），
避免构造需要真实模型的 ConversationAgent。上传/重建接口额外打桩以隔离文件系统与向量库。
"""
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
    # done 事件以 event: message 承载，type 在 data 的 JSON 中
    assert '"done"' in text


def test_diagnose_sse_stream(api_client):
    resp = api_client.post("/api/diagnose", json={"query": "设备无法启动"})
    assert resp.status_code == 200
    assert "event: message" in resp.text
    assert '"done"' in resp.text


def test_knowledge_upload(api_client, tmp_path, monkeypatch):
    # 把知识库落盘目录重定向到临时目录，避免污染项目 data/
    monkeypatch.setattr("api.routes.knowledge.get_abs_path", lambda p: str(tmp_path))
    resp = api_client.post(
        "/api/knowledge/upload",
        files={"files": ("test.txt", b"hello knowledge", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["file_count"] == 1
    assert os.path.exists(os.path.join(str(tmp_path), "test.txt"))


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
