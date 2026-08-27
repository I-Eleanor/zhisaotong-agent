"""Orchestrator（多 Agent 编排层）。

统一入口：根据意图把用户请求路由到「对话 Agent」或「诊断 Agent」。

路由策略（轻量、可兜底）：
    1. 关键词命中（不工作 / 故障 / 报错 / 异响 …）→ 诊断 Agent
    2. 未命中关键词 → 轻量 LLM 分类（conversation / diagnostic）
    3. LLM 不可用或失败 → 默认对话 Agent

对外仅提供同步 execute()：API 层统一通过 SSE bridge（线程 + 有界队列）
桥接到 FastAPI 异步路由，避免同一链路存在多套异步方案。
"""
import threading
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from agent.conversation_agent import ConversationAgent
from agent.diagnostic_agent import get_diagnostic_agent
from agent.events import (
    EVENT_TYPE_DONE,
    EVENT_TYPE_ERROR,
    EVENT_TYPE_HANDOFF_CREATED,
    EVENT_TYPE_HANDOFF_HUMAN_SERVICE,
    EVENT_TYPE_HANDOFF_SUGGESTED,
    AgentEvent,
    make_event,
)
from utils.logger_handler import log_safe_text, logger, safe_exception_fields
from utils.prompt_loader import load_orchestrator_prompt
from utils.request_context import get_request_id

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

# 用户显式转人工关键词：优先级高于一切路由（即使 mode 已指定也拦截）。
# 刻意不收录单字"人工"，避免"人工智能"等无关误触发。
HANDOFF_KEYWORDS = ["转人工", "人工客服", "联系人工", "转接人工", "我要人工"]

HANDOFF_SUGGEST_MESSAGE = (
    "已收到您的转人工请求，正在为您创建人工工单，请稍候……"
)

HANDOFF_HUMAN_SERVICE_MESSAGE = (
    "您当前已转人工服务，新消息将实时同步给客服并由此回复，请您稍候。"
)

HANDOFF_CREATED_MESSAGE = (
    "人工工单已创建成功，客服将尽快与您联系。"
)

HANDOFF_CREATE_FAILED_MESSAGE = (
    "人工工单创建失败，请稍后重试或直接致电客服。"
)


def match_handoff_keyword(user_query: str) -> str | None:
    """命中转人工关键词时返回该关键词，否则返回 None。"""
    q = (user_query or "").lower()
    return next((kw for kw in HANDOFF_KEYWORDS if kw.lower() in q), None)


class Orchestrator:
    def __init__(self, conversation_agent: ConversationAgent | None = None, diagnostic_agent=None,
                 handoff_tickets=None):
        """conversation_agent / diagnostic_agent 均可注入（应用容器管理的依赖）；
        未注入时回退各自的默认构造（保持旧调用方兼容）。

        handoff_tickets：可选的工单仓库（应用容器注入 HandoffStore 的轻量代理），
        提供 create() 并在命中转人工关键词时创建工单；未注入时（纯单元测试 /
        旧调用方）仍输出 handoff_suggested 事件，不创建工单。
        """
        self.conversation_agent = conversation_agent or ConversationAgent()
        self.diagnostic_agent = diagnostic_agent or get_diagnostic_agent()
        self.handoff_tickets = handoff_tickets

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
            logger.warning({"event": "route_llm_failed", "fallback": MODE_CONVERSATION, **safe_exception_fields(e)})
            return MODE_CONVERSATION

    # ----------------------------------------------------------- 执行入口

    def execute(
        self,
        user_query: str,
        history: list | None = None,
        mode: str | None = None,
        conversation_id: str | None = None,
    ) -> Iterator[AgentEvent]:
        """统一同步执行入口，返回 SSE/UI 可消费的 AgentEvent 流。

        用户显式转人工：最高优先级，先于一切路由（含显式 mode）。
        命中关键词时若注入了 handoff_tickets 仓库，则创建工单并产出
        handoff_created 事件；仓库未配置/创建失败时产出 handoff_suggested
        或 error 事件（见 _run_handoff）。

        conversation_id：前端会话内稳定的会话标识。若提供，则人工接管检测
        优先使用它定位活动工单；未提供时回退到从 history 推导。
        """
        # 会话标识：显式提供优先；否则从 history 推导作为弱关联
        effective_cid = conversation_id or self._conversation_id(history)

        # 用户显式转人工：最高优先级，先于一切路由（含显式 mode）
        keyword = match_handoff_keyword(user_query)
        if keyword:
            yield from self._run_handoff(user_query, history, keyword, conversation_id=effective_cid)
            return

        # 人工接管期间：模型停手，不参与回复；用户新消息写入工单实时对话日志，
        # 由人工在后台看到并回复；确认完成工单（resolved/closed）后模型才重新介入。
        if not effective_cid:
            effective_cid = user_query[:64]
        if self.handoff_tickets is not None and effective_cid:
            active = self.handoff_tickets.get_active_by_conversation(effective_cid)
            if active is not None:
                self.handoff_tickets.append_user_message(effective_cid, user_query.strip())
                logger.info({
                    "event": "handoff_human_service",
                    "ticket_id": active.get("ticket_id"),
                    "conversation_id": effective_cid,
                    "request_id": get_request_id(),
                })
                yield make_event(
                    EVENT_TYPE_HANDOFF_HUMAN_SERVICE,
                    agent="orchestrator",
                    content=HANDOFF_HUMAN_SERVICE_MESSAGE,
                )
                yield make_event(EVENT_TYPE_DONE, agent="orchestrator")
                return

        effective_mode = mode or self.route(user_query)
        logger.info({"event": "orchestrator_execute", "mode": effective_mode, "query": log_safe_text(user_query)})

        if effective_mode == MODE_DIAGNOSTIC:
            yield from self.diagnostic_agent.run(user_query)
        else:
            yield from self.conversation_agent.stream(user_query, history)

    # ----------------------------------------------------------- 转人工闭环

    def _run_handoff(self, user_query: str, history: list | None, keyword: str,
                     conversation_id: str | None = None) -> Iterator[AgentEvent]:
        """命中转人工关键词：先报 handoff_suggested，再尝试创建工单。

        事件顺序：handoff_suggested → (handoff_created | error) → done。
        - 仓库已注入且创建成功：handoff_created（含 ticket_id / status / message，以及一次性凭证 access_key，仅此事件下发一次）；
        - 仓库未注入（纯单元测试/旧调用方）：只发 handoff_suggested（兼容旧行为）；
        - 仓库注入但创建失败：error 事件（统一错误，不泄漏内部细节）。
        """
        logger.info({
            "event": "handoff_suggested",
            "keyword": keyword,
            "query": log_safe_text(user_query),
            "stage": "orchestrator",
        })
        yield make_event(
            EVENT_TYPE_HANDOFF_SUGGESTED,
            agent="orchestrator",
            content=HANDOFF_SUGGEST_MESSAGE,
            reason="user_request",
            keyword=keyword,
        )

        if self.handoff_tickets is None:
            # 未配置工单仓库：仅提示（保持无存储环境下行为兼容）
            yield make_event(EVENT_TYPE_DONE, agent="orchestrator")
            return

        try:
            result = self.handoff_tickets.create(
                user_question=user_query.strip(),
                conversation_id=conversation_id or self._conversation_id(history),
                recent_conversations=history or [],
                conversation_summary=self._conversation_summary(history),
                retrieval_sources=None,
                diagnostic_result="",
                handoff_reason="user_request",
            )
        except Exception as e:
            logger.error({
                "event": "handoff_ticket_create_failed",
                "keyword": keyword,
                "stage": "orchestrator",
                "error_type": type(e).__name__,
            })
            yield make_event(
                EVENT_TYPE_ERROR,
                agent="orchestrator",
                content=HANDOFF_CREATE_FAILED_MESSAGE,
                error_code="HANDOFF_TICKET_CREATE_FAILED",
            )
            yield make_event(EVENT_TYPE_DONE, agent="orchestrator")
            return

        if result is None:
            # 工单仓库未配置（如 HANDOFF_DB_PATH 未设置）：只提示，不产出 created
            logger.warning({
                "event": "handoff_ticket_unavailable",
                "keyword": keyword,
                "stage": "orchestrator",
            })
            yield make_event(EVENT_TYPE_DONE, agent="orchestrator")
            return

        ticket, access_key = result

        logger.info({
            "event": "handoff_created",
            "ticket_id": ticket.get("ticket_id"),
            "conversation_id": ticket.get("conversation_id"),
            "request_id": get_request_id(),
        })
        # 明文 access_key 仅在 handoff_created 事件中下发一次：日志、其他事件、
        # 普通消息与错误响应均不得携带。前端收到后仅在会话内存保存。
        yield make_event(
            EVENT_TYPE_HANDOFF_CREATED,
            agent="orchestrator",
            content=HANDOFF_CREATED_MESSAGE,
            ticket_id=ticket.get("ticket_id", ""),
            access_key=access_key,
            status=ticket.get("status", ""),
            message=HANDOFF_CREATED_MESSAGE,
        )
        yield make_event(EVENT_TYPE_DONE, agent="orchestrator")

    @staticmethod
    def _conversation_id(history: list | None) -> str | None:
        """从 history 中提取会话标识：优先取首条 user 消息（弱会话关联）。"""
        for message in history or []:
            if isinstance(message, dict) and message.get("role") == "user":
                return message.get("content", "")[:64]
        return None

    @staticmethod
    def _conversation_summary(history: list | None) -> str:
        """最近对话的文本化摘要：拼接 role + content，供人工客服快速浏览。"""
        parts = []
        for message in history or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "")
            content = message.get("content", "")
            if content:
                parts.append(f"{role}: {content}")
        return "\n".join(parts[-8:])


_orchestrator = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> Orchestrator:
    """全局懒加载单例（双检锁防并发首次请求重复构建 Agent）。"""
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
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
