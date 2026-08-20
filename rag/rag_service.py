import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import get_chat_model
from rag.reranker import create_reranker
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.prompt_loader import load_rag_prompts


def print_prompt(prompt):
    logger.debug({"event": "rag_prompt_render", "prompt": prompt.to_string()})
    return prompt


class RagSummarizeService:
    def __init__(self, vector_store: VectorStoreService = None, model=None):
        self.vector_store = vector_store or VectorStoreService()
        # 优先环境变量（本地开发用已下载模型），需验证路径存在
        # 路径不存在时（如 Docker 容器内）自动回退到 config 中的 HF 模型名
        _env_reranker = os.getenv("LOCAL_RERANKER_PATH")
        reranker_model = _env_reranker if (_env_reranker and os.path.exists(_env_reranker)) else chroma_conf.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker = create_reranker(
            enabled=chroma_conf.get("reranker_enabled", False),
            model_name=reranker_model,
            top_k=chroma_conf.get("reranker_top_k", 3),
        )
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = model or get_chat_model()
        self.chain = self._init_chain()
        self.confidence_threshold = chroma_conf.get("confidence_threshold", 0.3)

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str, source_files: list[str] | str | None = None) -> list[Document]:
        """带分数检索文档（relevance_score 写入 metadata）；传入 source_files 时限定检索范围。"""
        return self.vector_store.search_with_scores(query, source_files=source_files)

    # ------------------------------------------------- 统一检索链路

    def _retrieve_and_rerank(self, query: str, source_files: list[str] | str | None = None) -> list[Document]:
        """唯一检索链路：向量召回（带 relevance_score）→ CrossEncoder 重排（带 rerank_score）。

        build_context / rag_summarize / rag_with_sources 共用本方法，
        保证分数语义与置信度判断行为一致。
        """
        docs = self.retriever_docs(query, source_files=source_files)
        if not docs:
            logger.warning({"event": "rag_no_results", "query": query, "source_files": source_files})
            return []
        return self.reranker.rerank(query, docs)

    def _top_confidence(self, docs: list[Document]) -> float:
        """取首条文档的置信度：优先 rerank_score（重排信号更准），否则 relevance_score。

        分数缺失按 0 处理（低置信度），绝不默认高置信度。
        """
        if not docs:
            return 0.0
        top = docs[0].metadata
        if top.get("rerank_score") is not None:
            return float(top["rerank_score"])
        return float(top.get("relevance_score", 0.0))

    def _format_context(self, docs: list[Document]) -> str:
        """把重排后的文档拼装为带引用来源的上下文文本。"""
        context = ""
        for counter, doc in enumerate(docs, start=1):
            citation = self._format_citation(doc.metadata)
            context += f"【参考资料{counter}】来源：{citation}\n内容：{doc.page_content}\n\n"
        return context

    def _invoke_chain(self, query: str, context: str) -> str:
        """统一的 LLM 总结入口。"""
        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

    @staticmethod
    def _format_citation(metadata: dict) -> str:
        source_file = metadata.get("source_file", "未知来源")
        chunk_index = metadata.get("chunk_index", -1)
        page = metadata.get("page", -1)

        if page and page >= 0:
            return f"{source_file}，第{page}页，第{chunk_index}段"
        return f"{source_file}，第{chunk_index}段"

    NO_RESULT_MESSAGE = ("抱歉，我未检索到与您问题相关的可靠资料，无法给出准确回答。"
                         "建议您：1）换一种方式描述问题；2）咨询人工客服获取帮助。")

    LOW_CONFIDENCE_PREFIX = "⚠️ 知识库依据不足："

    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        """从检索结果中提取结构化来源信息（score 为向量相关性，rerank_score 为重排分数）。"""
        sources = []
        for doc in docs:
            sources.append({
                "document": doc.metadata.get("source_file", "未知来源"),
                "chunk_id": doc.metadata.get("chunk_index", -1),
                "score": round(doc.metadata.get("relevance_score", 0.0), 4),
                "rerank_score": round(doc.metadata["rerank_score"], 4) if doc.metadata.get("rerank_score") is not None else None,
            })
        return sources

    def build_context(self, query: str, source_files: list[str] | str | None = None) -> str:
        """执行「向量召回 → 重排 → 拼装带引用的上下文」，返回空字符串表示未检索到资料。"""
        context_docs = self._retrieve_and_rerank(query, source_files=source_files)

        if not context_docs:
            return ""

        context = self._format_context(context_docs)

        logger.info({
            "event": "rag_build_context",
            "query": query,
            "source_files": source_files,
            "doc_count": len(context_docs),
            "sources": [doc.metadata.get("source_file", "未知") for doc in context_docs],
        })

        return context

    def rag_summarize(self, query: str, source_files: list[str] | str | None = None) -> str:
        context_docs = self._retrieve_and_rerank(query, source_files=source_files)

        if not context_docs:
            return self.NO_RESULT_MESSAGE

        context = self._format_context(context_docs)
        confidence = self._top_confidence(context_docs)

        logger.info({
            "event": "rag_summarize",
            "query": query,
            "source_files": source_files,
            "confidence": confidence,
        })

        answer = self._invoke_chain(query, context)
        if confidence < self.confidence_threshold:
            answer = f"{self.LOW_CONFIDENCE_PREFIX}以下回答可能不够准确，建议进一步确认。\n\n{answer}"
        return answer

    def rag_with_sources(self, query: str, source_files: list[str] | str | None = None) -> dict:
        """RAG 问答并返回结构化来源信息。"""
        context_docs = self._retrieve_and_rerank(query, source_files=source_files)

        if not context_docs:
            return {
                "answer": self.NO_RESULT_MESSAGE,
                "sources": [],
                "confidence": 0.0,
            }

        context = self._format_context(context_docs)
        sources = self._extract_sources(context_docs)
        confidence = self._top_confidence(context_docs)

        logger.info({
            "event": "rag_with_sources",
            "query": query,
            "doc_count": len(context_docs),
            "sources": [s["document"] for s in sources],
            "confidence": confidence,
        })

        answer = self._invoke_chain(query, context)
        if confidence < self.confidence_threshold:
            answer = f"{self.LOW_CONFIDENCE_PREFIX}以下回答可能不够准确，建议进一步确认。\n\n{answer}"

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
