"""对话 Agent（ReAct 模式）。

由原 react_agent.py 重构而来，保留全部既有能力（7 个工具 + 三层中间件），
并新增：
1. 多轮对话记忆——execute_stream 支持传入 history，Agent 能理解上下文追问；
2. 结构化事件流——对外输出统一的 AgentEvent，供前端（React）/ SSE 分类渲染；
3. 依赖注入——model / tools / middleware 均可从外部传入，便于测试替换为 Mock。
"""
from typing import AsyncIterator, Iterator

from langchain.agents import create_agent

from agent.events import AgentEvent, make_event
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from model.factory import get_chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts

AGENT_NAME = "conversation"

DEFAULT_TOOLS = [rag_summarize, get_weather, get_user_location, get_user_id,
                 get_current_month, fetch_external_data, fill_context_for_report]

DEFAULT_MIDDLEWARE = [monitor_tool, log_before_model, report_prompt_switch]


class ConversationAgent:
    def __init__(self, model=None, tools: list = None, middleware: list = None, system_prompt: str = None):
        self.agent = create_agent(
            model=model or get_chat_model(),
            system_prompt=system_prompt or load_system_prompts(),
            tools=DEFAULT_TOOLS if tools is None else tools,
            middleware=DEFAULT_MIDDLEWARE if middleware is None else middleware,
        )

    # --------------------------------------------------------------- 输入构造

    @staticmethod
    def _build_input(query: str, history: list[dict] = None) -> dict:
        """把历史消息与本轮提问拼装为 Agent 输入。

        history 形如 [{"role": "system"|"user"|"assistant", "content": "..."}]，
        通常由 ConversationBuffer.get_history_for_query() 提供。
        """
        messages: list[dict] = []

        for message in history or []:
            role = message.get("role")
            content = message.get("content")
            if role in ("system", "user", "assistant") and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})
        return {"messages": messages}

    # --------------------------------------------------------------- 流式解析

    @staticmethod
    def _message_key(message) -> str:
        return getattr(message, "id", None) or f"{type(message).__name__}:{id(message)}"

    def _to_events(self, message, skip: set = None) -> list[AgentEvent]:
        """把 LangGraph 输出的单条消息翻译为对外事件。

        skip: 历史助手消息内容集合，用于跳过把历史回显成新事件（避免前端重复）。
        """
        events: list[AgentEvent] = []
        message_type = type(message).__name__

        if message_type == "HumanMessage":
            # 用户自己的输入，不回显
            return events

        if message_type == "ToolMessage":
            events.append(make_event(
                "tool_end",
                agent=AGENT_NAME,
                content=str(getattr(message, "content", "") or ""),
                tool=getattr(message, "name", ""),
            ))
            return events

        for tool_call in getattr(message, "tool_calls", None) or []:
            events.append(make_event(
                "tool_start",
                agent=AGENT_NAME,
                tool=tool_call.get("name", ""),
                args=tool_call.get("args", {}),
            ))

        content = getattr(message, "content", "")
        if isinstance(content, list):
            # 部分模型返回 content blocks，拼接其中的文本片段
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        content = (content or "").strip()

        if content:
            # 跳过历史助手消息的回显（其内容与历史完全一致）
            if skip and content in skip:
                return events
            events.append(make_event("message", agent=AGENT_NAME, content=content + "\n"))

        return events

    # --------------------------------------------------------------- 执行入口

    def stream(self, query: str, history: list[dict] = None) -> Iterator[AgentEvent]:
        """流式执行，产出结构化事件。"""
        input_dict = self._build_input(query, history)
        # 历史助手消息内容集合，用于跳过历史回显
        skip = {m.get("content", "").strip() for m in (history or []) if m.get("role") == "assistant"}

        logger.info({
            "event": "conversation_agent_start",
            "query": query,
            "history_messages": len(input_dict["messages"]) - 1,
        })

        seen: set[str] = set()

        try:
            # 第三个参数 context 即 runtime 上下文，是动态提示词切换的标记位
            for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
                for message in chunk.get("messages", []):
                    key = self._message_key(message)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield from self._to_events(message, skip)
        except Exception as e:
            logger.error({
                "event": "conversation_agent_error",
                "query": query,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            })
            yield make_event("error", agent=AGENT_NAME, content=f"对话处理失败：{str(e)}")

        yield make_event("done", agent=AGENT_NAME)

    async def astream(self, query: str, history: list[dict] = None) -> AsyncIterator[AgentEvent]:
        """异步流式执行，供 FastAPI SSE 使用。"""
        input_dict = self._build_input(query, history)
        # 历史助手消息内容集合，用于跳过历史回显
        skip = {m.get("content", "").strip() for m in (history or []) if m.get("role") == "assistant"}

        logger.info({
            "event": "conversation_agent_astart",
            "query": query,
            "history_messages": len(input_dict["messages"]) - 1,
        })

        seen: set[str] = set()

        try:
            async for chunk in self.agent.astream(input_dict, stream_mode="values", context={"report": False}):
                for message in chunk.get("messages", []):
                    key = self._message_key(message)
                    if key in seen:
                        continue
                    seen.add(key)
                    for event in self._to_events(message, skip):
                        yield event
        except Exception as e:
            logger.error({
                "event": "conversation_agent_error",
                "query": query,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            })
            yield make_event("error", agent=AGENT_NAME, content=f"对话处理失败：{str(e)}")

        yield make_event("done", agent=AGENT_NAME)

    def execute_stream(self, query: str, history: list[dict] = None) -> Iterator[str]:
        """兼容旧调用方式的纯文本流式接口。"""
        from agent.events import events_to_text
        yield from events_to_text(self.stream(query, history))

    def invoke(self, query: str, history: list[dict] = None) -> str:
        """一次性返回完整回答（取最后一条 message 事件）。"""
        answer = ""
        for event in self.stream(query, history):
            if event.get("type") in ("message", "error"):
                answer = event.get("content", "")
        return answer.strip()


if __name__ == '__main__':
    agent = ConversationAgent()

    for text in agent.execute_stream("给我生成我的使用报告"):
        print(text, end="", flush=True)
