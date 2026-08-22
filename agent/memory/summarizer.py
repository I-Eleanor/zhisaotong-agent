"""对话摘要生成器。

分层记忆的第二层：当对话轮数超过阈值时，把「溢出」的旧对话压缩成 2-3 句话摘要，
以极小的 token 代价保留远期上下文。

设计要点：
1. 摘要是「增量」的——每次都把上一次的摘要和新溢出的消息一起喂给 LLM，避免信息丢失；
2. LLM 调用失败时降级为纯文本截断，绝不因为摘要失败而中断主流程；
3. 通过 create_summarizer(enabled=False) 可完全关闭，测试与离线场景零 LLM 开销。
"""
from abc import ABC, abstractmethod

from utils.logger_handler import logger, safe_exception_fields

SUMMARY_PROMPT = """你是对话摘要助手。请把下面这段「扫地机器人客服」的历史对话压缩为简洁摘要。

要求：
1. 只用中文，控制在 {max_chars} 字以内，2-3 句话；
2. 必须保留：用户提到的机型/环境/故障现象、用户的核心诉求、已经给出的关键结论；
3. 保留代词指代所需的实体（例如「滤网」「主刷」「1001 号用户」），便于后续追问理解上下文；
4. 丢弃寒暄、重复表述和无信息量的客套话；
5. 直接输出摘要正文，不要任何前缀、标题或解释。

{previous_summary_block}
待压缩的历史对话：
{conversation}
"""


class BaseSummarizer(ABC):
    @abstractmethod
    def summarize(self, messages: list[dict], previous_summary: str = "") -> str:
        """把 messages 压缩为摘要文本，previous_summary 为已有摘要（增量压缩）。"""


class NoopSummarizer(BaseSummarizer):
    """不做摘要，直接丢弃溢出消息（保留已有摘要）。"""

    def summarize(self, messages: list[dict], previous_summary: str = "") -> str:
        return previous_summary


class ConversationSummarizer(BaseSummarizer):
    def __init__(self, model=None, max_chars: int = 300):
        self._model = model
        self.max_chars = max_chars

    @property
    def model(self):
        # 懒加载，避免导入即创建模型
        if self._model is None:
            from model.factory import get_chat_model
            self._model = get_chat_model()
        return self._model

    @staticmethod
    def _render_conversation(messages: list[dict]) -> str:
        role_label = {"user": "用户", "assistant": "客服", "system": "系统"}
        lines = []
        for message in messages:
            role = role_label.get(message.get("role", ""), message.get("role", ""))
            content = (message.get("content") or "").strip()
            if content:
                lines.append(f"{role}：{content}")
        return "\n".join(lines)

    def _fallback(self, conversation: str, previous_summary: str) -> str:
        """LLM 不可用时的降级方案：直接截断拼接，保证记忆链路不断。"""
        merged = f"{previous_summary}\n{conversation}".strip()
        if len(merged) > self.max_chars:
            merged = merged[-self.max_chars:]
        return merged

    def summarize(self, messages: list[dict], previous_summary: str = "") -> str:
        conversation = self._render_conversation(messages)
        if not conversation:
            return previous_summary

        previous_summary_block = (
            f"已有摘要（请在此基础上合并更新）：\n{previous_summary}\n" if previous_summary else ""
        )

        prompt = SUMMARY_PROMPT.format(
            max_chars=self.max_chars,
            previous_summary_block=previous_summary_block,
            conversation=conversation,
        )

        try:
            response = self.model.invoke(prompt)
            summary = getattr(response, "content", response)
            summary = str(summary).strip()

            if not summary:
                raise ValueError("摘要结果为空")

            logger.info({
                "event": "memory_summarized",
                "input_messages": len(messages),
                "summary_length": len(summary),
            })
            return summary
        except Exception as e:
            logger.error({
                "event": "memory_summarize_error",
                **safe_exception_fields(e),
            })
            return self._fallback(conversation, previous_summary)


def create_summarizer(enabled: bool = True, model=None, max_chars: int = 300) -> BaseSummarizer:
    if not enabled:
        logger.info({"event": "summarizer_init", "mode": "noop"})
        return NoopSummarizer()

    logger.info({"event": "summarizer_init", "mode": "llm", "max_chars": max_chars})
    return ConversationSummarizer(model=model, max_chars=max_chars)
