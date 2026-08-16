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
        self.retriever = self.vector_store.get_retriever()
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

    def retriever_docs(self, query: str, source_files: list[str] | str = None) -> list[Document]:
        """检索文档；传入 source_files 时把检索范围限定在指定知识库文件内。"""
        if source_files:
            retriever = self.vector_store.get_retriever(source_files=source_files)
            return retriever.invoke(query)
        return self.retriever.invoke(query)

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
        """从检索结果中提取结构化来源信息。"""
        sources = []
        for doc in docs:
            sources.append({
                "document": doc.metadata.get("source_file", "未知来源"),
                "chunk_id": doc.metadata.get("chunk_index", -1),
                "score": round(doc.metadata.get("relevance_score", 0.0), 4),
            })
        return sources

    def _check_confidence(self, docs: list[Document]) -> bool:
        """检查检索结果的置信度是否达到阈值。"""
        if not docs:
            return False
        top_score = docs[0].metadata.get("relevance_score", 1.0)
        return top_score >= self.confidence_threshold

    def build_context(self, query: str, source_files: list[str] | str = None) -> str:
        """执行「向量召回 → 重排 → 拼装带引用的上下文」，返回空字符串表示未检索到资料。"""
        context_docs = self.retriever_docs(query, source_files=source_files)

        if not context_docs:
            logger.warning({"event": "rag_no_results", "query": query, "source_files": source_files})
            return ""

        context_docs = self.reranker.rerank(query, context_docs)

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            citation = self._format_citation(doc.metadata)
            context += f"【参考资料{counter}】来源：{citation}\n内容：{doc.page_content}\n\n"

        logger.info({
            "event": "rag_build_context",
            "query": query,
            "source_files": source_files,
            "doc_count": len(context_docs),
            "sources": [doc.metadata.get("source_file", "未知") for doc in context_docs],
        })

        return context

    def rag_summarize(self, query: str, source_files: list[str] | str = None) -> str:
        context = self.build_context(query, source_files=source_files)

        if not context:
            return self.NO_RESULT_MESSAGE

        logger.info({"event": "rag_summarize", "query": query, "source_files": source_files})

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

    def rag_with_sources(self, query: str, source_files: list[str] | str = None) -> dict:
        """RAG 问答并返回结构化来源信息。"""
        context_docs = self.retriever_docs(query, source_files=source_files)

        if not context_docs:
            logger.warning({"event": "rag_no_results", "query": query, "source_files": source_files})
            return {
                "answer": self.NO_RESULT_MESSAGE,
                "sources": [],
                "confidence": 0.0,
            }

        context_docs = self.reranker.rerank(query, context_docs)
        sources = self._extract_sources(context_docs)
        top_score = sources[0]["score"] if sources else 0.0

        context = ""
        for counter, doc in enumerate(context_docs, start=1):
            citation = self._format_citation(doc.metadata)
            context += f"【参考资料{counter}】来源：{citation}\n内容：{doc.page_content}\n\n"

        logger.info({
            "event": "rag_with_sources",
            "query": query,
            "doc_count": len(context_docs),
            "sources": [s["document"] for s in sources],
            "top_score": top_score,
        })

        if top_score < self.confidence_threshold:
            answer = self.chain.invoke({"input": query, "context": context})
            answer = f"{self.LOW_CONFIDENCE_PREFIX}以下回答可能不够准确，建议进一步确认。\n\n{answer}"
        else:
            answer = self.chain.invoke({"input": query, "context": context})

        return {
            "answer": answer,
            "sources": sources,
            "confidence": top_score,
        }


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
