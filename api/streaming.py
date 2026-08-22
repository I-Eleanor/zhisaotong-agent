"""SSE 桥接工具：把同步事件生成器转成异步 SSE 流。

架构：FastAPI async route → SSE bridge（线程 + 有界队列）→ 同步 Agent generator。
（对话 Agent 的中间件目前只有同步版 wrap_tool_call，为复用已验证的同步 execute
路径，用线程 + 队列桥接到 EventSourceResponse，保持真正的流式输出。）

要点：
- 业务事件队列有界（默认 100），满时丢弃普通事件并告警（背压）；
- 终止信号（error / done）走独立的控制队列，与业务队列彻底分离，
  绝不争抢容量：任意 queue_maxsize（含 1）下错误通知与结束信号都永不丢失，
  流必然结束且不重复；
- contextvars.copy_context() 把 request_id 等上下文带进生产线程，
  保证 Agent / RAG / Tool 日志与请求使用同一 request_id；
- 心跳（SSE 注释行，前端与代理忽略）防止空闲断连；
- 客户端断开：置 stop 标志、停止读取并关闭同步生成器、限时回收 Future；
- 生产线程异常：完整异常只进日志，客户端只收到安全错误事件；
- 全程非阻塞投递（put_nowait + call_soon_threadsafe），生产线程永不卡死。
"""
import asyncio
import contextvars
import json
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

from sse_starlette.sse import EventSourceResponse

from agent.events import AgentEvent
from utils import error_codes
from utils.logger_handler import logger, safe_exception_fields
from utils.request_context import get_request_id, set_request_id

# 有界队列容量：慢消费者场景下超过容量的事件被丢弃并告警（背压）
SSE_QUEUE_MAXSIZE = 100
# 心跳间隔（秒）：空闲时发送 SSE 注释行
SSE_HEARTBEAT_SECONDS = 15.0
# 客户端断开后等待生产线程退出的时限（秒）
SSE_DRAIN_TIMEOUT = 5.0

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def get_sse_executor() -> ThreadPoolExecutor:
    """获取 SSE 线程池（懒创建；lifespan 未初始化时的兜底入口）。"""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="sse-bridge")
        return _executor


def shutdown_sse_executor(wait: bool = True) -> None:
    """回收 SSE 线程池（由 FastAPI lifespan 在应用关闭时调用）。"""
    global _executor
    with _executor_lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=wait)
        logger.info({"event": "sse_executor_shutdown"})


def build_sse_stream(
    sync_runner: Callable[[], Iterator[AgentEvent]],
    request_id: str = "",
    queue_maxsize: int = SSE_QUEUE_MAXSIZE,
    heartbeat_seconds: float = SSE_HEARTBEAT_SECONDS,
    drain_timeout: float = SSE_DRAIN_TIMEOUT,
) -> AsyncIterator[dict]:
    """构造 SSE 事件异步生成器；queue/心跳/回收参数可注入，便于测试。

    必须在请求上下文（request_id 已设置）中调用：上下文在此时复制并带进生产线程。
    """
    rid = request_id or get_request_id()
    # 复制当前异步上下文（含 request_id），确保线程内日志与请求同源
    context = contextvars.copy_context()

    async def event_generator() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        # 业务事件队列：有界，满时丢弃普通事件（背压），防止慢客户端内存无限增长
        queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        # 终止控制队列：与业务队列彻底分离，无界（至多 error + done 两项）。
        # error / done 绝不与普通事件争抢容量，任意 queue_maxsize（含 1）下都永不丢失
        control: asyncio.Queue = asyncio.Queue()
        # 唤醒信号：两队列皆空时消费端挂起等待；生产回调投递任意项后 set
        wakeup = asyncio.Event()
        stop = threading.Event()
        dropped_events: list[int] = []

        def schedule_event(event: AgentEvent) -> None:
            """普通业务事件 → 有界队列；满时丢弃并计数（仅普通事件可丢弃）。"""

            def _run() -> None:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    dropped_events.append(1)
                wakeup.set()

            loop.call_soon_threadsafe(_run)

        def schedule_signal(kind: str) -> None:
            """终止信号（error / done）→ 独立控制队列；非阻塞、永不丢弃。"""

            def _run() -> None:
                control.put_nowait(kind)
                wakeup.set()

            loop.call_soon_threadsafe(_run)

        def produce() -> None:
            # run_in_executor 不复制 ContextVar：线程入口显式设置 request_id（兜底）
            set_request_id(rid)
            gen: Iterator[AgentEvent] | None = None
            try:
                gen = sync_runner()
                for event in gen:
                    if stop.is_set():
                        break
                    schedule_event(event)
            except Exception as e:
                logger.error({
                    "event": "sse_producer_error",
                    "request_id": rid,
                    "stage": "stream",
                    **safe_exception_fields(e),
                })
                schedule_signal("error")
            finally:
                # 关闭同步生成器（触发其内部清理）；仅当其支持 close 时
                if gen is not None:
                    close = getattr(gen, "close", None)
                    if close is not None:
                        with suppress(Exception):
                            close()
                if dropped_events:
                    logger.warning({
                        "event": "sse_queue_dropped",
                        "request_id": rid,
                        "dropped": len(dropped_events),
                    })
                schedule_signal("done")

        fut = loop.run_in_executor(get_sse_executor(), context.run, produce)

        # 生产线程异常时下发的安全错误事件（原始异常只进日志，不发给客户端）
        safe_error_event = {
            "event": "error",
            "data": json.dumps(
                {
                    "type": "error",
                    "agent": "",
                    "content": "服务暂时异常，请稍后重试",
                    "data": {"error_code": error_codes.STREAM_FAILED, "request_id": rid},
                },
                ensure_ascii=False,
            ),
        }

        try:
            while True:
                # 1. 快路径：优先排空业务队列（所有普通事件先于 error 输出）
                try:
                    event = queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    yield {"event": "message", "data": json.dumps(event, ensure_ascii=False)}
                    continue

                # 2. 业务队列空：处理控制信号（error / done）
                try:
                    signal = control.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                else:
                    if signal == "error":
                        yield safe_error_event
                        continue
                    break  # done：流正常结束

                # 3. 两队列皆空：挂起等待新数据/终止（clear 后无需复查——
                #    同步段不会有回调插入，set 只能发生在 await 挂起期间）
                wakeup.clear()
                try:
                    await asyncio.wait_for(wakeup.wait(), timeout=heartbeat_seconds)
                except TimeoutError:
                    # 心跳：SSE 注释行，前端与代理均忽略
                    yield {"comment": "ping"}
                continue
        finally:
            # 客户端断开或流正常结束：停止生产、回收 Future（限时等待防止悬挂）
            stop.set()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(fut), timeout=drain_timeout)

    return event_generator()


def sse_bridge(sync_runner: Callable[[], Iterator[AgentEvent]], request_id: str = "") -> EventSourceResponse:
    """sync_runner() 返回同步事件生成器；返回 EventSourceResponse（SSE 流）。"""
    return EventSourceResponse(build_sse_stream(sync_runner, request_id=request_id))
