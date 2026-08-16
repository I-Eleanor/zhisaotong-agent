"""端到端冒烟测试：验证 3 条核心链路（使用 TestClient + Mock，不需要真实 LLM）。

1. /api/health 返回成功
2. 普通知识问答收到完整 SSE（start → message → done）
3. 故障问题路由到 Diagnostic Agent 并生成报告
"""
import contextlib
import json


def test_e2e_health(api_client):
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_e2e_knowledge_chat_sse_complete(api_client):
    """知识问答链路：收到完整 SSE 流（message → done）"""
    resp = api_client.post("/api/chat", json={"query": "滤网怎么换", "mode": "conversation"})
    assert resp.status_code == 200
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data:"):
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line[5:].strip()))
    types = [e.get("type") for e in events]
    assert any(t in types for t in ("message", "tool_start")), "应有业务事件"
    assert "done" in types, "应有 done 事件"


def test_e2e_diagnostic_route_and_report(api_client, monkeypatch):
    """故障问题路由到 Diagnostic Agent 并生成报告"""
    events_log = []

    class TrackingOrchestrator:
        def execute(self, query, history=None, mode=None):
            def gen():
                yield {"type": "route", "agent": "orchestrator", "content": "", "data": {"mode": "diagnostic", "mode_label": "设备诊断"}}
                events_log.append("route_diagnostic")
                yield {"type": "plan", "agent": "diagnostic", "content": "", "data": {"steps": ["检查电源", "检查电池"]}}
                events_log.append("plan")
                yield {"type": "step", "agent": "diagnostic", "content": "检查电源适配器连接", "data": {"index": 1, "description": "检查电源"}}
                events_log.append("step")
                yield {"type": "report", "agent": "diagnostic", "content": "电源适配器连接正常，建议检查电池触点。"}
                events_log.append("report")
                yield {"type": "done", "agent": "diagnostic", "content": ""}
                events_log.append("done")
            return gen()

    monkeypatch.setattr("api.routes.diagnostic.get_orchestrator", lambda: TrackingOrchestrator())
    resp = api_client.post("/api/diagnose", json={"query": "扫地机充不进电"})
    assert resp.status_code == 200
    assert "route_diagnostic" in events_log, "应路由到 diagnostic"
    assert "report" in events_log, "应生成诊断报告"
    assert "done" in events_log, "应以 done 结束"
