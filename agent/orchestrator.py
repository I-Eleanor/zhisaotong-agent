"""Orchestrator（多 Agent 编排层）。

统一入口：根据意图把用户请求路由到「对话 Agent」或「诊断 Agent」。

路由策略（轻量、可兜底）：
    1. 关键词命中（不工作 / 故障 / 报错 / 异响 …）→ 诊断 Agent
    2. 未命中关键词 → 轻量 LLM 分类（conversation / diagnostic）
    3. LLM 不可用或失败 → 默认对话 Agent

对外提供同步 execute()（供前端（React）/ 测试）与异步 aexecute()（供 Phase 2 SSE）。
"""
from collections.abc import AsyncIterator, Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from agent.conversation_agent import ConversationAgent
from agent.diagnostic_agent import get_diagnostic_agent
from agent.events import AgentEvent
from utils.logger_handler import logger
from utils.prompt_loader import load_orchestrator_prompt

DIAGNOSTIC_KEYWORDS = [
    "不工作", "故障", "报错", "异响", "不转", "不亮", "漏水", "卡住", "异常",
    "不吸", "不动", "停机", "死机", "充不进", "不回充", "离线", "掉线",
    "error", "fault", "stuck", "dead",
]

# 知识问询类模式：即使命中诊断关键词，也应优先走对话 Agent
# 例："常见故障有哪些" → 对话 Agent（知识问答），而非诊断 Agent（执行排查）
KNOWLEDGE_PATTERNS = [
    "常见", "有哪些", "是什么", "什么原因", "为什么", "怎么回事",
    "如何", "介绍", "了解", "讲解", "说明", "解释",
    "区别", "对比", "优缺点", "推荐", "选购", "保养",
    "清理", "清洁", "更换", "安装", "使用", "维护", "重置",
]

MODE_CONVERSATION: str = "conversation"
MODE_DIAGNOSTIC: str = "diagnostic"
VALID_MODES: frozenset[str] = frozenset({MODE_CONVERSATION, MODE_DIAGNOSTIC})
RouteResult = str


class Orchestrator:
    def __init__(self, conversation_agent: ConversationAgent = None):
        self.conversation_agent = conversation_agent or ConversationAgent()
        self.diagnostic_agent = get_diagnostic_agent()

    # ----------------------------------------------------------- 意图路由

    def route(self, user_query: str) -> RouteResult:
        """返回 MODE_CONVERSATION 或 MODE_DIAGNOSTIC。"""
        q = (user_query or "").lower()

        # 先检查是否为知识问询（即使命中诊断关键词也优先对话）
        for pat in KNOWLEDGE_PATTERNS:
            if pat in q:
                logger.info({"event": "route_knowledge_pattern", "pattern": pat, "mode": MODE_CONVERSATION})
                return MODE_CONVERSATION

        for kw in DIAGNOSTIC_KEYWORDS:
            if kw.lower() in q:
                logger.info({"event": "route_keyword", "keyword": kw, "mode": MODE_DIAGNOSTIC})
                return MODE_DIAGNOSTIC

        # 关键词未命中 → 轻量 LLM 分类
        mode = self._llm_classify(user_query)
        logger.info({"event": "route_llm", "mode": mode})
        return mode

    def _llm_classify(self, user_query: str) -> RouteResult:
        try:
            from model.factory import get_chat_model
            model = get_chat_model()
            resp = model.invoke([
                SystemMessage(content=load_orchestrator_prompt()),
                HumanMessage(content=user_query or ""),
            ])
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            if "diagnos" in content.lower() or MODE_DIAGNOSTIC in content.lower():
                return MODE_DIAGNOSTIC
            return MODE_CONVERSATION
        except Exception as e:
            logger.warning({"event": "route_llm_failed", "error": str(e), "fallback": MODE_CONVERSATION})
            return MODE_CONVERSATION

    # ----------------------------------------------------------- 执行入口

    def execute(self, user_query: str, history: list = None, mode: str = None) -> Iterator[AgentEvent]:
        """统一同步执行入口，返回 SSE/UI 可消费的 AgentEvent 流。"""
        effective_mode = mode or self.route(user_query)
        logger.info({"event": "orchestrator_execute", "mode": effective_mode, "query": user_query})

        if effective_mode == MODE_DIAGNOSTIC:
            yield from self.diagnostic_agent.run(user_query)
        else:
            yield from self.conversation_agent.stream(user_query, history)

    async def aexecute(self, user_query: str, history: list = None, mode: str = None) -> AsyncIterator[AgentEvent]:
        """统一异步执行入口（供 FastAPI SSE 使用）。"""
        effective_mode = mode or self.route(user_query)
        logger.info({"event": "orchestrator_aexecute", "mode": effective_mode, "query": user_query})

        if effective_mode == MODE_DIAGNOSTIC:
            for ev in self.diagnostic_agent.run(user_query):
                yield ev
        else:
            async for ev in self.conversation_agent.astream(user_query, history):
                yield ev


_orchestrator = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    global _orchestrator
    _orchestrator = None


if __name__ == '__main__':
    orch = Orchestrator()
    for ev in orch.execute("扫地机不工作了", mode=None):
        print(ev)
    print("---")
    for ev in orch.execute("怎么清理滤网"):
        print(ev)
