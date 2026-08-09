"""分层多轮对话记忆。

第一层（近期）：完整保留最近 N 轮对话，原样进入 messages；
第二层（远期）：超出 N 轮的旧对话交给 Summarizer 压缩成摘要，
                以一条 system message 的形式前置注入。

一「轮」= 一条 user 消息 + 其后紧邻的 assistant 回复。
只有形成完整轮次的消息才会被计入压缩范围，避免把用户刚发出、
还没得到回复的那条消息误压缩掉。
"""
from utils.config_handler import agent_conf
from utils.logger_handler import logger
from agent.memory.summarizer import BaseSummarizer, create_summarizer

SUMMARY_MESSAGE_TEMPLATE = "以下是你与该用户更早之前对话的摘要，请结合它理解用户当前的问题：\n{summary}"


class ConversationBuffer:
    def __init__(self, max_rounds: int = None, summarizer: BaseSummarizer = None, summary_enabled: bool = None):
        memory_conf = (agent_conf or {}).get("memory", {}) or {}

        self.max_rounds = max_rounds if max_rounds is not None else memory_conf.get("max_rounds", 5)
        if self.max_rounds < 1:
            self.max_rounds = 1

        if summary_enabled is None:
            summary_enabled = memory_conf.get("summary_enabled", True)

        self._summarizer = summarizer or create_summarizer(
            enabled=summary_enabled,
            max_chars=memory_conf.get("summary_max_chars", 300),
        )

        self._messages: list[dict] = []
        self._summary: str = ""

    # ------------------------------------------------------------------ 写入

    def add_message(self, role: str, content: str) -> None:
        if not content:
            return
        self._messages.append({"role": role, "content": content})
        self._compress_if_needed()

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_message("assistant", content)

    def clear(self) -> None:
        self._messages = []
        self._summary = ""

    # ------------------------------------------------------------------ 读取

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def recent_messages(self) -> list[dict]:
        """未被压缩的近期原始消息。"""
        return list(self._messages)

    def get_messages(self) -> list[dict]:
        """返回可直接喂给 Agent 的消息列表：[摘要 system 消息] + 近 N 轮完整消息。"""
        messages: list[dict] = []

        if self._summary:
            messages.append({
                "role": "system",
                "content": SUMMARY_MESSAGE_TEMPLATE.format(summary=self._summary),
            })

        messages.extend(self._messages)
        return messages

    def get_history_for_query(self) -> list[dict]:
        """获取「不含当前这条用户提问」的历史消息，供 Agent 拼接使用。

        典型用法：先 add_user_message(query)，再取 history 传给 Agent，
        此时需要把刚加进去的那条 user 消息排除掉，避免重复。
        """
        messages = self.get_messages()
        if messages and messages[-1].get("role") == "user":
            return messages[:-1]
        return messages

    # ------------------------------------------------------------------ 压缩

    def _count_complete_rounds(self) -> int:
        rounds = 0
        index = 0
        while index < len(self._messages) - 1:
            if self._messages[index]["role"] == "user" and self._messages[index + 1]["role"] == "assistant":
                rounds += 1
                index += 2
            else:
                index += 1
        return rounds

    def _split_overflow(self) -> tuple[list[dict], list[dict]]:
        """按完整轮次切分：返回 (需要压缩的溢出消息, 需要保留的近期消息)。"""
        boundaries: list[int] = []  # 每个完整轮次结束后的下标
        index = 0
        while index < len(self._messages) - 1:
            if self._messages[index]["role"] == "user" and self._messages[index + 1]["role"] == "assistant":
                boundaries.append(index + 2)
                index += 2
            else:
                index += 1

        overflow_round_count = len(boundaries) - self.max_rounds
        if overflow_round_count <= 0:
            return [], self._messages

        cut = boundaries[overflow_round_count - 1]
        return self._messages[:cut], self._messages[cut:]

    def _compress_if_needed(self) -> None:
        if self._count_complete_rounds() <= self.max_rounds:
            return

        overflow, remaining = self._split_overflow()
        if not overflow:
            return

        self._summary = self._summarizer.summarize(overflow, previous_summary=self._summary)
        self._messages = remaining

        logger.info({
            "event": "memory_compressed",
            "compressed_messages": len(overflow),
            "remaining_messages": len(remaining),
            "max_rounds": self.max_rounds,
            "has_summary": bool(self._summary),
        })
