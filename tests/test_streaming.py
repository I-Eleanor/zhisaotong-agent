"""SSE 桥接测试（P0-3）。

覆盖场景（对应修改指令四.8）：
- request_id 跨线程保持一致（contextvars 传递）
- 生成器异常产生 error 事件（安全信息，不泄漏原始异常）
- 客户端断开后停止生产（stop 标志 + 生成器关闭）
- 慢消费者不会导致队列无限增长（有界队列背压）
- 多个并发请求不串 request_id
- 空闲期发送心跳
- lifespan 关闭时回收线程池
"""
import asyncio
import json
import time

from api.streaming import build_sse_stream, get_sse_executor
from utils.request_context import get_request_id, request_id_var


def _collect(stream, timeout: float = 10.0) -> list[dict]:
    """同步消费一个 SSE 异步生成器，返回事件列表。"""

    async def run():
        out = []
        async for item in stream:
            out.append(item)
        return out

    return asyncio.run(asyncio.wait_for(run(), timeout))


# ----------------------------------------------------------------- request_id 跨线程
def test_request_id_crosses_thread_boundary():
    """生产线程内的 get_request_id() 与请求上下文一致。"""
    seen = []

    def runner():
        seen.append(get_request_id())
        yield {"type": "message", "content": "hi"}
        yield {"type": "done"}

    token = request_id_var.set("rid-cross-123")
    try:
        out = _collect(build_sse_stream(runner))
    finally:
        request_id_var.reset(token)

    assert seen == ["rid-cross-123"], "生产线程应看到与请求一致的 request_id"
    assert any('"hi"' in item["data"] for item in out)


def test_request_id_falls_back_to_current_context():
    """未显式传 request_id 时，使用调用时的上下文值。"""
    seen = []

    def runner():
        seen.append(get_request_id())
        yield {"type": "done"}

    token = request_id_var.set("rid-auto-456")
    try:
        _collect(build_sse_stream(runner, request_id=""))
    finally:
        request_id_var.reset(token)

    assert seen == ["rid-auto-456"]


# ----------------------------------------------------------------- 异常安全
def test_producer_exception_yields_single_safe_error_event():
    """生产线程异常：恰好一个 error 事件，内容为安全信息，不含原始异常文本。"""

    def runner():
        yield {"type": "message", "content": "ok"}
        raise RuntimeError("敏感内部错误 boom")

    out = _collect(build_sse_stream(runner, request_id="rid-err"))
    kinds = [item.get("event") for item in out]
    assert kinds.count("error") == 1, "error 事件应恰好出现一次（不重复）"
    err = next(item for item in out if item.get("event") == "error")
    data = json.loads(err["data"])
    assert data["type"] == "error"
    assert data["data"]["error_code"] == "STREAM_FAILED"
    assert data["data"]["request_id"] == "rid-err"
    assert "boom" not in data["content"], "不得向客户端泄漏原始异常文本"
    # 异常前的事件仍正常送达，流正常结束（收集完成即生成器耗尽）
    assert any('"ok"' in item["data"] for item in out)


# ----------------------------------------------------------------- 客户端断开
def test_client_disconnect_stops_producer():
    """客户端断开（提前关闭流）后，生产线程应停止生产。"""
    produced = []

    def runner():
        for i in range(100):
            produced.append(i)
            yield {"type": "message", "content": str(i)}
            time.sleep(0.005)

    async def consume_two():
        stream = build_sse_stream(runner, request_id="rid-disc")
        out = []
        async for item in stream:
            out.append(item)
            if len(out) >= 2:
                break
        await stream.aclose()  # 模拟客户端断开
        return out

    out = asyncio.run(asyncio.wait_for(consume_two(), timeout=10))
    assert len(out) == 2
    time.sleep(0.3)  # 留时间让 stop 标志生效
    assert len(produced) < 100, f"断开后生产者应停止（实际生产 {len(produced)}）"


# ----------------------------------------------------------------- 背压
def test_bounded_queue_drops_events_for_slow_consumer():
    """慢消费者：有界队列满后丢弃事件，消费数远小于生产数（内存不无限增长）。"""
    total = 300

    def runner():
        for i in range(total):
            yield {"type": "message", "content": str(i)}

    async def slow_consume():
        stream = build_sse_stream(
            runner, request_id="rid-slow", queue_maxsize=10, heartbeat_seconds=30.0,
        )
        out = []
        async for item in stream:
            out.append(item)
            await asyncio.sleep(0.002)  # 慢消费者
        return out

    out = asyncio.run(asyncio.wait_for(slow_consume(), timeout=30))
    assert len(out) < total, f"慢消费者应触发丢弃背压（收到 {len(out)} / 生产 {total}）"
    assert len(out) > 0, "仍应收到部分事件"


# ----------------------------------------------------------------- 并发隔离
def test_concurrent_streams_keep_separate_request_ids():
    """并发流各自的 request_id 互不串扰。"""

    def make_runner(tag: str):
        def runner():
            yield {"type": "message", "content": f"{tag}:{get_request_id()}"}
            yield {"type": "done"}

        return runner

    async def run_one(tag: str, rid: str):
        token = request_id_var.set(rid)
        try:
            stream = build_sse_stream(make_runner(tag))
            out = []
            async for item in stream:
                out.append(item)
            return out
        finally:
            request_id_var.reset(token)

    async def main():
        return await asyncio.gather(run_one("a", "rid-a"), run_one("b", "rid-b"))

    out_a, out_b = asyncio.run(asyncio.wait_for(main(), timeout=15))
    assert "a:rid-a" in out_a[0]["data"], "流 A 的事件应携带 rid-a"
    assert "b:rid-b" in out_b[0]["data"], "流 B 的事件应携带 rid-b"


# ----------------------------------------------------------------- 心跳
def test_heartbeat_emitted_while_producer_idle():
    """生产者长时间无事件时，消费端收到心跳注释行。"""

    def runner():
        yield {"type": "message", "content": "first"}
        time.sleep(0.3)  # 生产间隔超过心跳周期
        yield {"type": "done"}

    out = _collect(build_sse_stream(runner, request_id="rid-hb", heartbeat_seconds=0.05))
    pings = [item for item in out if "comment" in item]
    assert pings, "空闲期应发送心跳注释行"
    assert pings[0]["comment"] == "ping"
    # 业务事件不受心跳影响
    assert any('"first"' in item["data"] for item in out)


# ----------------------------------------------------------------- lifespan
def test_lifespan_shuts_down_sse_executor():
    """with TestClient 退出（应用关闭）时，SSE 线程池被回收。"""
    from fastapi.testclient import TestClient

    from api import streaming
    from api.main import app
    from tests.conftest import CannedOrchestrator

    try:
        with TestClient(app) as client:
            # lifespan 已创建容器：把容器内 orchestrator 换成桩，避免构造真实 Agent
            app.state.container._orchestrator = CannedOrchestrator()
            resp = client.post("/api/chat", json={"query": "hi"})
            assert resp.status_code == 200
            assert streaming._executor is not None, "请求后线程池应已创建"
    finally:
        del app.state.container  # 清理挂载，避免污染后续测试

    assert streaming._executor is None, "应用关闭（lifespan shutdown）时应回收线程池"
    # 关闭后再次使用：懒重建兜底，服务不因关闭而永久失效
    assert get_sse_executor() is not None
    streaming.shutdown_sse_executor(wait=True)


def test_events_use_message_event_type():
    """SSE 事件类型保持向后兼容：message / error / done。"""

    def runner():
        yield {"type": "message", "agent": "conversation", "content": "你好"}
        yield {"type": "done", "agent": "conversation", "content": ""}

    out = _collect(build_sse_stream(runner, request_id="rid-compat"))
    kinds = [item.get("event") for item in out]
    assert kinds == ["message", "message"], "AgentEvent 应以 message 事件类型下发"
    first = json.loads(out[0]["data"])
    assert first["type"] == "message"


# ----------------------------------------------------------------- 关键事件不丢失（结束信号）
def _consume_after_warmup(stream, warmup_seconds: float = 0.5) -> list[dict]:
    """先取一个事件启动生产线程，等待队列填满（稳态）后再开始消费。

    用于构造确定性场景：生产线程跑完后，有界队列被普通事件填满，
    stop/error 等关键事件必须在满队情况下仍能送达。
    """

    async def consume():
        ait = stream.__aiter__()
        out = [await ait.__anext__()]  # 启动生成器与生产线程，取走第一个事件
        await asyncio.sleep(warmup_seconds)  # 生产线程跑完：队列填满、溢出事件被丢弃
        async for item in ait:
            out.append(item)
        return out

    return asyncio.run(asyncio.wait_for(consume(), timeout=5))


def test_stop_signal_survives_full_queue():
    """队列被普通事件填满后，stop 结束信号仍能送达，流自然结束（不靠超时强杀）。"""
    total = 100

    def runner():
        for i in range(total):
            yield {"type": "message", "content": str(i)}

    stream = build_sse_stream(
        runner, request_id="rid-stop-full", queue_maxsize=3, heartbeat_seconds=30.0,
    )
    # wait_for 超时会抛 TimeoutError：流必须在 5s 内自然结束（stop 送达）
    out = _consume_after_warmup(stream)
    messages = [item for item in out if item.get("event") == "message"]
    assert len(messages) > 0, "应收到部分事件"


def test_error_event_survives_full_queue_on_producer_failure():
    """队列满时生产器抛异常：error 通知与结束信号都必须送达，流正常结束。"""

    def runner():
        for i in range(50):
            yield {"type": "message", "content": str(i)}
        raise RuntimeError("内部敏感错误 secret-boom")

    stream = build_sse_stream(
        runner, request_id="rid-err-full", queue_maxsize=2, heartbeat_seconds=30.0,
    )
    out = _consume_after_warmup(stream)
    errors = [item for item in out if item.get("event") == "error"]
    assert len(errors) == 1, f"队列满时 error 事件也绝不能丢（收到 {len(errors)} 个）"
    data = json.loads(errors[0]["data"])
    assert data["data"]["error_code"] == "STREAM_FAILED"
    assert "secret-boom" not in data["content"], "不得向客户端泄漏原始异常文本"


def test_overflow_normal_events_dropped_when_queue_full():
    """普通事件超出队列容量时允许丢弃（背压），收到的 message 数远小于生产数。"""
    total = 200

    def runner():
        for i in range(total):
            yield {"type": "message", "content": str(i)}

    stream = build_sse_stream(
        runner, request_id="rid-drop-full", queue_maxsize=3, heartbeat_seconds=30.0,
    )
    out = _consume_after_warmup(stream)
    messages = [item for item in out if item.get("event") == "message"]
    assert 0 < len(messages) < total, (
        f"超出容量的普通事件应被丢弃（收到 {len(messages)}/{total}），且流正常结束"
    )


def test_streams_always_terminate_with_end_signal():
    """正常与空生成器：流必定在限定时间内自然结束（结束信号必达）。"""

    def empty_runner():
        yield from []

    def normal_runner():
        yield {"type": "message", "content": "x"}
        yield {"type": "done"}

    # 空流：无任何业务事件，仍必须收到 stop 并结束（_collect 超时会抛异常）
    out_empty = _collect(build_sse_stream(empty_runner, request_id="rid-empty"), timeout=5)
    assert out_empty == []

    out_norm = _collect(build_sse_stream(normal_runner, request_id="rid-norm"), timeout=5)
    assert len(out_norm) == 2, "两条事件后流应自然结束"


# ------------------------------------------------- 容量 1：error/done 与业务事件不争抢
def test_capacity_one_error_then_terminate_on_producer_failure():
    """queue_maxsize=1：快速生产多个普通事件后抛异常。

    断言：流限时结束、恰好一个 error、error 不含原始异常文本、error 是最后一个事件。
    """
    total = 20

    def runner():
        for i in range(total):
            yield {"type": "message", "content": str(i)}
        raise RuntimeError("内部敏感错误 secret-boom")

    stream = build_sse_stream(
        runner, request_id="rid-cap1-err", queue_maxsize=1, heartbeat_seconds=30.0,
    )
    out = _consume_after_warmup(stream, warmup_seconds=0.5)

    errors = [item for item in out if item.get("event") == "error"]
    assert len(errors) == 1, f"容量 1 时 error 也必须恰好送达一次（实际 {len(errors)}）"
    data = json.loads(errors[0]["data"])
    assert data["data"]["error_code"] == "STREAM_FAILED"
    assert data["data"]["request_id"] == "rid-cap1-err"
    assert "secret-boom" not in data["content"], "不得向客户端泄漏原始异常文本"
    assert out[-1]["event"] == "error", "error 之后不应再有事件，流随即正常结束"

    # 业务队列容量 1：慢消费窗口内仅能缓冲少量普通事件，绝大多数被背压丢弃
    messages = [item for item in out if item.get("event") == "message"]
    assert 1 <= len(messages) < total, (
        f"容量 1 的业务队列应触发大量丢弃（收到 {len(messages)}/{total}），"
        "但 error 与结束信号不受影响"
    )


def test_capacity_one_empty_producer_ends_immediately():
    """queue_maxsize=1 + 空生产器：无任何事件，流立即正常结束。"""

    def runner():
        yield from []

    out = _collect(
        build_sse_stream(runner, request_id="rid-cap1-empty", queue_maxsize=1),
        timeout=5,
    )
    assert out == [], "空生产器应立即结束（无 message / error / ping）"


# ----------------------------------------------------------------- 异常日志收敛（P1-6）
def test_producer_error_log_uses_safe_summary_no_traceback(caplog):
    """SSE 生产线程异常：日志用统一摘要（无 traceback / exc_info / 原始密钥），错误事件行为不变。"""
    import logging

    def runner():
        yield {"type": "message", "content": "ok"}
        raise RuntimeError("SSE 生产失败：sk-SSE-998877665544 密钥泄漏")

    with caplog.at_level(logging.ERROR, logger="agent"):
        out = _collect(build_sse_stream(runner, request_id="rid-sse-log"))

    # 客户侧行为不变：恰好一个安全 error 事件，流正常结束
    kinds = [item.get("event") for item in out]
    assert kinds.count("error") == 1
    err_data = json.loads(next(i for i in out if i.get("event") == "error")["data"])
    assert err_data["data"]["error_code"] == "STREAM_FAILED"
    assert "sk-SSE-998877665544" not in err_data["content"]

    entries = [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "sse_producer_error"
    ]
    assert entries, "应记录结构化错误日志"
    entry = entries[-1]
    assert entry["request_id"] == "rid-sse-log", "保留原有结构化上下文字段"
    assert entry["stage"] == "stream"
    assert entry["error_type"] == "RuntimeError"
    assert "sk-SSE-998877665544" not in entry["error_msg"], "原始密钥不得进日志"
    assert "traceback" not in entry, "不得记录 traceback 字段"

    record = next(r for r in caplog.records
                  if isinstance(r.msg, dict) and r.msg.get("event") == "sse_producer_error")
    assert record.exc_info is None, "禁止 exc_info=True"
