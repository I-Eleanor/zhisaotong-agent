"""SSE 桥接工具：把同步事件生成器转成异步 SSE 流。

原因：对话 Agent 的中间件目前只有同步版 wrap_tool_call，在异步 astream
上下文会报 “Asynchronous implementation of awrap_tool_call is not available”。
为复用第一阶段已验证可用的同步 execute 路径，这里用线程 + 队列把同步生成器
桥接到 EventSourceResponse，既保持真正的流式输出，又避免改动 LangChain 中间件
的异步协议细节。
"""
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from sse_starlette.sse import EventSourceResponse

_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="sse-bridge")


def sse_bridge(sync_runner):
    """sync_runner() 返回同步生成器，产出 AgentEvent 字典。

    返回 EventSourceResponse，内部用线程驱动同步生成器，经 asyncio 队列
    桥接到异步 SSE 流；客户端断开时通过 threading.Event 尽早中止生产者。
    """

    async def event_generator():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def produce():
            try:
                for event in sync_runner():
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
            except Exception as e:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("stop", None))

        fut = loop.run_in_executor(_executor, produce)
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "stop":
                    break
                if kind == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            {"type": "error", "content": f"出错：{payload}"}, ensure_ascii=False
                        ),
                    }
                    continue
                yield {"event": "message", "data": json.dumps(payload, ensure_ascii=False)}
        finally:
            stop.set()
        await fut

    return EventSourceResponse(event_generator())
