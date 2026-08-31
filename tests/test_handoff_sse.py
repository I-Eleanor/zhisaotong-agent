"""人工转接事件的 SSE 与同步接口透传测试。

覆盖（对应"完善 SSE 透传"要求）：
- 顺序：handoff_suggested → handoff_created → done（含失败场景 error）
- SSE 线缆格式与 AgentEvent 完全兼容（统一 event: message 帧 + JSON）
- handoff_created 事件内含一次性 access_key，且其余事件（suggested/done）与哈希均不含
- 有界队列满时 handoff_created 不丢失、不被打乱顺序
- 客户端断开：生产线程/队列/资源正常清理
- /chat/sync 人工转接响应 {answer, need_handoff: true}
- /chat/sync 普通问题响应兼容 {answer, need_handoff: false}
"""
import asyncio
import json
import time

from agent.events import make_event
from tests.conftest import CannedOrchestrator


def _collect(stream, timeout: float = 10.0) -> list[dict]:
    async def run():
        out = []
        async for item in stream:
            out.append(item)
        return out

    return asyncio.run(asyncio.wait_for(run(), timeout))


def _decode_types(out: list[dict]) -> list[str]:
    """从 SSE 帧中解出 AgentEvent 的 type 序列（仅 message 帧为业务事件）。"""
    return [
        json.loads(item["data"])["type"]
        for item in out if item.get("event") == "message"
    ]


def _handoff_runner():
    yield make_event("handoff_suggested", agent="orchestrator", content="正在创建工单",
                     reason="user_request", keyword="转人工")
    yield make_event("handoff_created", agent="orchestrator", content="已创建工单",
                     ticket_id="ticket-sse-1", access_key="sse-pat-key-12345678901234567890",
                     status="pending", message="人工工单已创建")
    yield make_event("done", agent="orchestrator")


# ------------------------------------------------------------- 事件顺序
def test_handoff_events_emitted_in_order_via_sse():
    """handoff_suggested → handoff_created → done 按序透传到 SSE。"""
    from api.streaming import build_sse_stream

    out = _collect(build_sse_stream(_handoff_runner, request_id="rid-order"))
    assert _decode_types(out) == ["handoff_suggested", "handoff_created", "done"], (
        "SSE 应严格按 suggested → created → done 顺序输出"
    )


def test_handoff_failure_order_suggested_error_done():
    """建单失败：handoff_suggested → error → done 按序透传。"""
    from api.streaming import build_sse_stream

    def runner():
        yield make_event("handoff_suggested", agent="orchestrator", content="正在创建工单")
        yield make_event("error", agent="orchestrator", content="创建失败",
                         error_code="HANDOFF_TICKET_CREATE_FAILED")
        yield make_event("done", agent="orchestrator")

    out = _collect(build_sse_stream(runner, request_id="rid-order-err"))
    assert _decode_types(out) == ["handoff_suggested", "error", "done"]
    created = json.loads(out[1]["data"])
    assert created["data"]["error_code"] == "HANDOFF_TICKET_CREATE_FAILED"


def test_handoff_events_complete_with_done():
    """normal 流：message + done 兼容，顺序与业务语义一致。"""
    from api.streaming import build_sse_stream

    def runner():
        yield make_event("message", agent="conversation", content="你好")
        yield make_event("done", agent="conversation")

    out = _collect(build_sse_stream(runner, request_id="rid-compat-order"))
    types = _decode_types(out)
    assert types == ["message", "done"]


# ------------------------------------------------------------- SSE 线缆格式
def test_sse_wire_format_is_message_frames():
    """所有 AgentEvent（含 handoff）统一以 event: message 帧 + UTF-8 JSON 下发。"""
    from api.streaming import build_sse_stream

    out = _collect(build_sse_stream(_handoff_runner, request_id="rid-frame"))
    assert all(item.get("event") == "message" for item in out), (
        "AgentEvent 必须以 message 事件类型传输（兼容既有前端）"
    )
    for item in out:
        data = json.loads(item["data"])
        assert {"type", "agent", "content"} <= set(data.keys()), "事件须为完整 AgentEvent"


# ------------------------------------------------------------- 凭证隔离
def test_handoff_created_contains_access_key_once():
    """明文 access_key 仅在 handoff_created 事件出现一次，其余事件与哈希不含。"""
    from api.streaming import build_sse_stream

    out = _collect(build_sse_stream(_handoff_runner, request_id="rid-key"))
    created = json.loads(out[1]["data"])
    assert created["type"] == "handoff_created"
    assert created["data"]["ticket_id"] == "ticket-sse-1"
    assert created["data"]["access_key"] == "sse-pat-key-12345678901234567890"
    assert created["data"]["status"] == "pending"
    assert created["data"]["message"]

    # access_key 仅出现在 created 帧：suggested / done 帧、以及全部原始文本的计数均为 1 次
    for item in out:
        if json.loads(item["data"])["type"] != "handoff_created":
            assert "access_key" not in item["data"], "非 created 事件不得携带凭证"
    raw = "\n".join(item.get("data") or "" for item in out)
    assert raw.count("sse-pat-key-" ) == 1, "明文 access_key 在整个流中只能出现一次"
    assert "access_key_hash" not in raw


# ------------------------------------------------------------- 队列满不丢失
def _consume_after_warmup(stream, warmup_seconds: float = 0.5) -> list[dict]:
    async def consume():
        ait = stream.__aiter__()
        out = [await ait.__anext__()]
        await asyncio.sleep(warmup_seconds)
        async for item in ait:
            out.append(item)
        return out

    return asyncio.run(asyncio.wait_for(consume(), timeout=5))


def test_handoff_created_survives_full_queue_in_order():
    """业务队列被普通事件填满并丢弃后：handoff_created 仍恰好送达一次且顺序在 done 前。"""
    from api.streaming import build_sse_stream

    def runner():
        for i in range(60):
            yield make_event("message", agent="conversation", content=str(i))
        yield make_event("handoff_created", agent="orchestrator", content="已创建工单",
                         ticket_id="ticket-t1", access_key="qfull-key-12345678901234567890",
                         status="pending", message="已创建工单")
        yield make_event("done", agent="orchestrator")

    out = _consume_after_warmup(build_sse_stream(
        runner, request_id="rid-full-created", queue_maxsize=3, heartbeat_seconds=30.0,
    ))
    types = _decode_types(out)
    created = [t for t in types if t == "handoff_created"]
    assert len(created) == 1, f"队列满时 handoff_created 绝不能丢（收到 {len(created)} 个）"
    assert types[-2:] == ["handoff_created", "done"], f"顺序应 …created→done（实际尾部 {types[-3:]}）"
    # 背压确实发生过：普通 message 远少于生产数
    assert types.count("message") < 60, "超容量的普通事件应被丢弃"

    payload = json.loads(out[-2]["data"])
    assert payload["data"]["ticket_id"] == "ticket-t1"
    # 即使业务队列满了，凭证仍随 created 事件完整到达
    assert payload["data"]["access_key"] == "qfull-key-12345678901234567890"


# ------------------------------------------------------------- 客户端断开清理
def test_client_disconnect_cleans_up_handoff_producer():
    """handoff 流中客户端提前断开：生产线程停止，队列/生成器回收。"""
    from api.streaming import build_sse_stream

    produced = []

    def runner():
        for i in range(100):
            produced.append(i)
            yield make_event("handoff_created", agent="orchestrator", content=str(i),
                             ticket_id=f"t{i}", status="pending", message="x")
            time.sleep(0.005)

    async def consume_two_and_disconnect():
        stream = build_sse_stream(runner, request_id="rid-disc-handoff")
        out = []
        async for item in stream:
            out.append(item)
            if len(out) >= 2:
                break
        await stream.aclose()
        return out

    out = asyncio.run(asyncio.wait_for(consume_two_and_disconnect(), timeout=10))
    assert len(out) == 2
    time.sleep(0.3)
    assert len(produced) < 100, f"断开后生产线程应停止（实际生产 {len(produced)}）"


# ------------------------------------------------------------- /chat/sync
class _HandoffOnlyOrchestrator:
    def execute(self, query, history=None, mode=None, conversation_id=None):
        yield make_event("handoff_suggested", agent="orchestrator", content="正在创建工单")
        yield make_event("handoff_created", agent="orchestrator", content="已创建工单",
                         ticket_id="ticket-sync-1", status="pending", message="已创建工单")
        yield make_event("done", agent="orchestrator")


def test_chat_sync_handoff_response(api_client):
    from api.main import app

    app.state.container._orchestrator = _HandoffOnlyOrchestrator()
    resp = api_client.post("/api/chat/sync", json={"query": "转人工"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "", "转人工场景无文本回答"
    assert body["need_handoff"] is True
    assert body["conversation_id"].startswith("conv-"), "新增会话标识字段（旧客户端可忽略）"
    assert body["user_id"] == "1001", "未传 user_id 时使用默认演示用户"


def test_chat_sync_normal_response_unchanged(api_client):
    from api.main import app

    app.state.container._orchestrator = CannedOrchestrator()
    resp = api_client.post("/api/chat/sync", json={"query": "你好"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["need_handoff"] is False
    assert body["answer"], "普通问答 answer 与原行为一致"
    # 原有两字段保持不变，新增会话字段（conversation_id / user_id）
    assert {"answer", "need_handoff"} <= set(body), "原有响应字段不变"
    assert set(body) == {"answer", "need_handoff", "conversation_id", "user_id"}, (
        "新增会话标识字段，旧客户端可忽略"
    )
