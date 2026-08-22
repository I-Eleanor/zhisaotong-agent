"""轻量应用容器：统一管理应用级资源的创建与释放。

设计要点：
- 全量持有：chat_model / embedding_model / vector_store / rag_service / orchestrator
  均由容器自身创建并独占持有（不经模块级全局 getter），两个容器互不共享资源；
- 懒加载 + 双检锁：资源首次访问时才创建（应用启动不加载模型 / Embedding / Chroma），
  并发首次访问只创建一次（同一把 RLock；属性构造期访问其他属性同线程可重入）；
- orchestrator 用容器管理的依赖组装：模型 / RAG / 知识工具全部注入，
  新 API 请求路径不依赖 get_orchestrator 等模块级全局 getter（旧入口仍可用）；
- 生命周期状态机 OPEN → CLOSING → CLOSED（P1-15）：CLOSING 期间资源访问
  抛 ContainerStateError（不产生游离于生命周期之外的新资源），CLOSED 后访问
  重新懒加载并回到 OPEN；并发 close() 经 Condition 等待，不重复释放；
- close() 幂等：依赖逆序释放容器持有的资源（无 close() 的跳过、单项失败
  不阻断其余且不外抛、finally 保证状态收敛到 CLOSED），不触碰全局单例；
- FastAPI lifespan 启动时新建容器挂到 app.state；get_app_container 是路由的
  Depends 入口，未挂载容器（未进入 lifespan 且未注入）时返回明确 503，
  绝不回退进程级全局容器、绝不隐式创建资源。
"""
from __future__ import annotations

import threading
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from utils import error_codes
from utils.exceptions import ContainerStateError
from utils.logger_handler import logger, safe_exception_fields
from utils.request_context import get_request_id

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models.chat_models import BaseChatModel

    from agent.orchestrator import Orchestrator
    from rag.rag_service import RagSummarizeService
    from rag.vector_store import VectorStoreService


class ContainerState(Enum):
    """容器生命周期状态。

    状态机：OPEN → CLOSING → CLOSED；CLOSED 后首次资源访问重新懒加载
    并回到 OPEN（显式重建）。不变式：CLOSING / CLOSED 时所有资源引用
    必为 None（引用清空与状态迁移在同一锁内原子完成）。
    """

    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


# CLOSING 期间资源访问的拒绝信息（五个懒加载属性共用；只进服务端日志）
_CLOSING_ACCESS_MSG = (
    "容器正在关闭中（CLOSING），拒绝创建新资源；"
    "请等待关闭完成（CLOSED 后访问将重新懒加载）"
)


def _log_container_not_ready(state: str) -> None:
    """container_not_ready 结构化日志：只含安全字段，不含异常原文。"""
    logger.warning({
        "event": "container_not_ready",
        "state": state,
        "request_id": get_request_id(),
        "error_code": error_codes.CONTAINER_NOT_READY,
    })


class AppContainer:
    """应用级资源容器：chat_model / embedding_model / vector_store / rag_service / orchestrator。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # 并发 close() 协调：后到的 close 在此等待进行中的关闭完成
        self._close_done = threading.Condition(self._lock)
        self._chat_model: BaseChatModel | None = None
        self._embedding_model: Embeddings | None = None
        self._vector_store: VectorStoreService | None = None
        self._rag_service: RagSummarizeService | None = None
        self._orchestrator: Orchestrator | None = None
        self._state = ContainerState.OPEN

    # ------------------------------------------- 懒加载资源（双检锁，线程安全）
    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            with self._lock:
                if self._state is ContainerState.CLOSING:
                    raise ContainerStateError(_CLOSING_ACCESS_MSG)
                if self._chat_model is None:
                    from model.factory import ChatModelFactory
                    self._chat_model = ChatModelFactory().generator()
                    self._state = ContainerState.OPEN
        return self._chat_model

    @property
    def embedding_model(self) -> Embeddings:
        if self._embedding_model is None:
            with self._lock:
                if self._state is ContainerState.CLOSING:
                    raise ContainerStateError(_CLOSING_ACCESS_MSG)
                if self._embedding_model is None:
                    from model.factory import EmbeddingsFactory
                    self._embedding_model = EmbeddingsFactory().generator()
                    self._state = ContainerState.OPEN
        return self._embedding_model

    @property
    def vector_store(self) -> VectorStoreService:
        if self._vector_store is None:
            with self._lock:
                if self._state is ContainerState.CLOSING:
                    raise ContainerStateError(_CLOSING_ACCESS_MSG)
                if self._vector_store is None:
                    from rag.vector_store import VectorStoreService
                    # 注入容器自己的 Embedding：向量库与容器其他资源同源，
                    # 两个容器各自持有独立的向量库实例
                    self._vector_store = VectorStoreService(embedding_function=self.embedding_model)
                    self._state = ContainerState.OPEN
        return self._vector_store

    @property
    def rag_service(self) -> RagSummarizeService:
        if self._rag_service is None:
            with self._lock:
                if self._state is ContainerState.CLOSING:
                    raise ContainerStateError(_CLOSING_ACCESS_MSG)
                if self._rag_service is None:
                    from rag.rag_service import RagSummarizeService
                    self._rag_service = RagSummarizeService(
                        vector_store=self.vector_store,
                        model=self.chat_model,
                    )
                    self._state = ContainerState.OPEN
        return self._rag_service

    @property
    def orchestrator(self) -> Orchestrator:
        if self._orchestrator is None:
            with self._lock:
                if self._state is ContainerState.CLOSING:
                    raise ContainerStateError(_CLOSING_ACCESS_MSG)
                if self._orchestrator is None:
                    self._orchestrator = self._build_orchestrator()
                    self._state = ContainerState.OPEN
        return self._orchestrator

    def _build_orchestrator(self) -> Orchestrator:
        """用容器管理的依赖组装 Orchestrator：模型、RAG、知识工具全部注入。"""
        from agent.conversation_agent import ConversationAgent
        from agent.diagnostic.parser import LlmParser
        from agent.diagnostic.service import DiagnosticAgent
        from agent.diagnostic.tool_router import ToolRouter
        from agent.knowledge_agent import KnowledgeAgent
        from agent.orchestrator import Orchestrator
        from agent.tools.agent_tools import build_conversation_tools

        knowledge = KnowledgeAgent(rag_service=self.rag_service)
        conversation_agent = ConversationAgent(
            model=self.chat_model,
            tools=build_conversation_tools(knowledge_agent=knowledge),
        )
        diagnostic_agent = DiagnosticAgent(
            parser=LlmParser(model=self.chat_model),
            tool_router=ToolRouter(knowledge_agent=knowledge),
            model=self.chat_model,
        )
        return Orchestrator(conversation_agent=conversation_agent, diagnostic_agent=diagnostic_agent)

    # ------------------------------------------- 生命周期
    @property
    def state(self) -> ContainerState:
        """当前生命周期状态（OPEN / CLOSING / CLOSED）。"""
        return self._state

    @property
    def closed(self) -> bool:
        return self._state is ContainerState.CLOSED

    def close(self) -> None:
        """释放容器管理的资源（幂等、并发安全；应用关闭时由 lifespan 调用）。

        并发语义（P1-15 状态机 OPEN → CLOSING → CLOSED）：
        - 状态迁移与引用清空在同一锁内原子完成：CLOSING 一旦可见，所有
          引用必为 None，懒加载属性因此不会返回半关闭资源；
        - CLOSING 期间新的资源访问抛 ContainerStateError——不会产生游离于本次
          生命周期之外的新资源（旧实现允许此时重建并把 closed 改回
          False，正是本任务修复的竞态）；
        - 创建与 close 快照共用同一把锁：close 等锁期间创建的资源必然
          被快照接管并释放，不泄漏；
        - 并发 close()：后到者在 Condition 上等待进行中的关闭完成再返回
          （任何 close() 返回即 CLOSED），不重复释放；
        - CLOSED 后首次资源访问重新懒加载回到 OPEN（显式重建）。

        释放语义（P1-14 保持不变）：
        - 依赖逆序释放：orchestrator → rag_service → vector_store →
          embedding_model → chat_model（先关上层消费者，再关底层被依赖资源）；
        - 逐项调用资源自身的 close()（若存在）；没有 close() 的资源仅清空
          引用、不抛异常；
        - 单个资源关闭失败：记录结构化日志（safe_exception_fields 统一形态）
          后继续释放其余资源，异常不向外抛出，避免遮蔽应用退出流程；
        - finally 保证无论释放过程是否出错，状态最终收敛到 CLOSED；
        - close() 在锁外调用（不持锁执行外部代码）；重复 close() 因引用已
          清空而幂等；
        - 不调用全局 reset_models() / reset_orchestrator() /
          reset_diagnostic_agent()——那些单例可能仍被未容器化的旧入口持有，
          不属于本容器的所有权范围。
        """
        with self._lock:
            while self._state is ContainerState.CLOSING:
                self._close_done.wait()  # 并发 close：等待进行中的关闭完成
            if self._state is ContainerState.CLOSED:
                return  # 幂等：已关闭（前次 close 已完成）
            self._state = ContainerState.CLOSING
            resources: list[tuple[str, object]] = [
                ("orchestrator", self._orchestrator),
                ("rag_service", self._rag_service),
                ("vector_store", self._vector_store),
                ("embedding_model", self._embedding_model),
                ("chat_model", self._chat_model),
            ]
            self._orchestrator = None
            self._rag_service = None
            self._vector_store = None
            self._embedding_model = None
            self._chat_model = None
        try:
            for name, resource in resources:
                if resource is None:
                    continue
                release = getattr(resource, "close", None)
                if release is None:
                    continue
                try:
                    release()
                except Exception as e:
                    logger.warning({
                        "event": "container_resource_close_failed",
                        "resource": name,
                        **safe_exception_fields(e),
                    })
        finally:
            with self._lock:
                self._state = ContainerState.CLOSED
                self._close_done.notify_all()
            logger.info({"event": "container_closed"})


def get_mounted_container(request: Request) -> AppContainer:
    """FastAPI Depends 入口：返回挂载的容器（只检查挂载，不检查状态）。

    供就绪探针使用：探针需要区分「未挂载」与「CLOSING / CLOSED」并
    返回自己的安全结构（status / checks），状态判断由探针自身完成。
    未挂载时记 container_not_ready 日志后抛 503，不回退全局容器。
    """
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        _log_container_not_ready("unmounted")
        raise HTTPException(status_code=503, detail="服务尚未就绪：应用容器未初始化")
    return container


def get_app_container(request: Request) -> AppContainer:
    """FastAPI Depends 入口（业务路由用）：返回挂载且处于 OPEN 状态的容器。

    容器由 lifespan 挂载到 app.state；测试环境未运行 lifespan 时由测试夹具
    显式注入。未挂载或非 OPEN（CLOSING / CLOSED）时记 container_not_ready
    日志后返回明确 503，不回退任何进程级全局容器、绝不隐式创建资源——
    尤其是容器关闭后到达的业务请求，不得经懒加载隐式重建资源
    （重建只能显式发生，游离于生命周期之外的资源会泄漏）。
    """
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        _log_container_not_ready("unmounted")
        raise HTTPException(status_code=503, detail="服务尚未就绪：应用容器未初始化")
    if container.state is not ContainerState.OPEN:
        _log_container_not_ready(container.state.value)
        raise HTTPException(status_code=503, detail="服务尚未就绪：应用容器已关闭")
    return container
