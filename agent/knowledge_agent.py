"""知识库 Agent。

架构模式：无 Agent 循环，纯 RAG 检索管道（向量召回 top-k → CrossEncoder 重排 top-n → LCEL 生成）。
定位：作为「被调用方」为对话 Agent 与诊断 Agent 提供统一的知识检索服务，
      对上层屏蔽向量库、重排器、提示词模板等实现细节。

相比直接使用 RagSummarizeService，本 Agent 额外提供：
1. 定向检索（限定知识库文件），供诊断 Agent 精准查故障手册 / 维护手册；
2. 懒加载，导入本模块不会触发 Embedding 模型与向量库初始化；
3. 统一的结构化日志与异常兜底。
"""
from langchain_core.documents import Document

from rag.rag_service import RagSummarizeService
from utils.logger_handler import log_safe_text, logger, safe_exception_fields

# 知识库文件名常量，供诊断 Agent 做定向检索
KB_TROUBLESHOOTING = "故障排除.txt"
KB_MAINTENANCE = "维护保养.txt"
KB_BUYING_GUIDE = "选购指南.txt"


class KnowledgeAgent:
    def __init__(self, rag_service: RagSummarizeService | None = None):
        self._rag = rag_service

    @property
    def rag(self) -> RagSummarizeService:
        if self._rag is None:
            self._rag = RagSummarizeService()
        return self._rag

    def retrieve_strict(self, query: str, source_files: list[str] | str | None = None) -> str:
        """检索并总结；异常直接抛出。

        供需要区分成功/失败的调用方（诊断 Agent 工具路由）使用，
        避免异常字符串被包装成正常检索结果。
        """
        result = self.rag.rag_summarize(query, source_files=source_files)
        logger.info({
            "event": "knowledge_agent_retrieve",
            "query": log_safe_text(query),
            "source_files": source_files,
            "result_length": len(result or ""),
        })
        return result

    def retrieve(self, query: str, source_files: list[str] | str | None = None) -> str:
        """检索并总结，返回带引用来源的自然语言答案（异常兜底为安全提示）。"""
        try:
            return self.retrieve_strict(query, source_files=source_files)
        except Exception as e:
            logger.error({
                "event": "knowledge_agent_error",
                "query": log_safe_text(query),
                "source_files": source_files,
                **safe_exception_fields(e),
            })
            # 异常细节只进日志，客户端只收到安全提示（该结果会进入对话 Agent 的回答链路）
            return "知识库检索暂时不可用，请稍后重试"

    def retrieve_docs(self, query: str, source_files: list[str] | str | None = None) -> list[Document]:
        """只做检索不做生成，返回原始文档（供需要自行拼装上下文的场景使用）。"""
        try:
            return self.rag.retriever_docs(query, source_files=source_files)
        except Exception as e:
            logger.error({
                "event": "knowledge_agent_retrieve_docs_error",
                "query": log_safe_text(query),
                **safe_exception_fields(e),
            })
            return []

    def retrieve_context_strict(self, query: str, source_files: list[str] | str | None = None) -> str:
        """返回「召回 + 重排 + 带引用拼装」后的上下文文本（不经过 LLM 总结）；异常直接抛出。

        诊断 Agent 的 Executor 使用它——把原始资料交给 Reporter 统一汇总，
        避免每一步都额外消耗一次 LLM 调用。
        """
        return self.rag.build_context(query, source_files=source_files)

    def retrieve_context(self, query: str, source_files: list[str] | str | None = None) -> str:
        """同 retrieve_context_strict，但异常兜底为空字符串（旧调用方兼容）。"""
        try:
            return self.retrieve_context_strict(query, source_files=source_files)
        except Exception as e:
            logger.error({
                "event": "knowledge_agent_context_error",
                "query": log_safe_text(query),
                **safe_exception_fields(e),
            })
            return ""

    # --------------------------------------------------------------- 定向检索

    def search_troubleshooting_strict(self, query: str) -> str:
        """在故障排除手册中定向检索；异常直接抛出。"""
        return self.retrieve_context_strict(query, source_files=KB_TROUBLESHOOTING)

    def search_troubleshooting(self, query: str) -> str:
        """在故障排除手册中定向检索（异常兜底为空字符串）。"""
        return self.retrieve_context(query, source_files=KB_TROUBLESHOOTING)

    def search_maintenance_strict(self, query: str) -> str:
        """在维护保养手册中定向检索；异常直接抛出。"""
        return self.retrieve_context_strict(query, source_files=KB_MAINTENANCE)

    def search_maintenance(self, query: str) -> str:
        """在维护保养手册中定向检索（异常兜底为空字符串）。"""
        return self.retrieve_context(query, source_files=KB_MAINTENANCE)


_knowledge_agent: KnowledgeAgent | None = None


def get_knowledge_agent() -> KnowledgeAgent:
    """获取全局知识库 Agent（懒加载单例，向量库与 Embedding 只初始化一次）。"""
    global _knowledge_agent
    if _knowledge_agent is None:
        _knowledge_agent = KnowledgeAgent()
    return _knowledge_agent


def reset_knowledge_agent() -> None:
    """重置单例，供测试隔离使用。"""
    global _knowledge_agent
    _knowledge_agent = None


if __name__ == '__main__':
    agent = KnowledgeAgent()
    print(agent.retrieve("小户型适合哪些扫地机器人"))
    print("-" * 40)
    print(agent.search_troubleshooting("吸力下降"))
