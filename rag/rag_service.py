import math
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import get_chat_model
from rag.reranker import create_reranker
from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf
from utils.logger_handler import log_safe_text, log_safe_value, logger
from utils.prompt_loader import load_rag_prompts


def _parse_relevance_score(raw) -> float:
    """把 metadata 中的 relevance_score 解析为 [0, 1] 的置信度分值。

    仅接受数值（bool 除外——Python 中 bool 是 int 子类，但不是合法分数）；
    None / 字符串 / 布尔 / NaN / 正负无穷等非法值一律返回 0.0
    （按低置信度处理），不抛异常、不默认高置信度。
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    score = float(raw)
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _parse_optional_display_score(raw) -> float | None:
    """把 metadata 中的展示分（如 rerank_score）解析为 [0, 1] 的四位小数值。

    bool / 字符串 / None / NaN / 正负无穷返回 None（保证 JSON 可安全序列化，
    不产生非标准的 NaN / Infinity 字面量）；合法数字钳制到 [0, 1] 并保留四位小数。
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    score = float(raw)
    if math.isnan(score) or math.isinf(score):
        return None
    return round(max(0.0, min(1.0, score)), 4)


def print_prompt(prompt):
    logger.debug({"event": "rag_prompt_render", "prompt": log_safe_text(prompt.to_string())})
    return prompt


class RagSummarizeService:
    def __init__(self, vector_store: VectorStoreService | None = None, model=None):
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

        relevance_score 专用于置信度判断；rerank_score 仅决定排序顺序并随来源透出展示。
        build_context / rag_summarize / rag_with_sources 共用本方法，
        保证分数语义与置信度判断行为一致。
        """
        docs = self.retriever_docs(query, source_files=source_files)
        if not docs:
            logger.warning({
                "event": "rag_no_results",
                "query": log_safe_text(query),
                "source_files": log_safe_value(source_files),
            })
            return []
        return self.reranker.rerank(query, docs)

    def _top_confidence(self, docs: list[Document]) -> float:
        """置信度 = 重排后第一篇文档的向量相关性（relevance_score），完整精度。

        置信度判断只看 relevance_score——它与 confidence_threshold 同源
        （均为向量检索相关性）；rerank_score 仅用于排序与展示，绝不参与
        是否拒答的决策：两类分数来源、分布和校准方式不同，不能共用阈值。
        分数缺失 / 非法 / NaN / 无穷一律按 0.0（低置信度）。
        返回值不取整：内部阈值判断必须用完整精度（0.29996 < 0.3 应判低置信度），
        四位小数只是 API 展示值（rag_with_sources / _extract_sources 负责取整），
        展示精度不得干扰业务判断。
        """
        if not docs:
            return 0.0
        return _parse_relevance_score(docs[0].metadata.get("relevance_score"))

    def _format_context(self, docs: list[Document]) -> str:
        """把重排后的文档拼装为带引用来源的上下文文本。"""
        context = ""
        for counter, doc in enumerate(docs, start=1):
            citation = self._format_citation(doc.metadata)
            context += f"【参考资料{counter}】来源：{citation}\n内容：{doc.page_content}\n\n"
        return context

    def _invoke_chain(self, query: str, context: str) -> str:
        """统一的 LLM 总结入口。"""
        return str(self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        ))

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
        """提取结构化来源：score 为向量相关性（与置信度同源，同为四位小数），
        rerank_score 为 CrossEncoder 展示分（sigmoid 归一化，未校准，仅排序与展示用）。

        rerank_score 经 _parse_optional_display_score 清洗，非法值（bool / 字符串 /
        None / NaN / 无穷）输出 None，保证 JSON 可安全序列化。
        """
        sources = []
        for doc in docs:
            sources.append({
                "document": doc.metadata.get("source_file", "未知来源"),
                "chunk_id": doc.metadata.get("chunk_index", -1),
                "score": round(_parse_relevance_score(doc.metadata.get("relevance_score")), 4),
                "rerank_score": _parse_optional_display_score(doc.metadata.get("rerank_score")),
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
            "query": log_safe_text(query),
            "source_files": log_safe_value(source_files),
            "doc_count": len(context_docs),
            "sources": log_safe_value([doc.metadata.get("source_file", "未知") for doc in context_docs]),
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
            "query": log_safe_text(query),
            "source_files": log_safe_value(source_files),
            "confidence": round(confidence, 4),
        })

        answer = self._invoke_chain(query, context)
        if confidence < self.confidence_threshold:
            answer = f"{self.LOW_CONFIDENCE_PREFIX}以下回答可能不够准确，建议进一步确认。\n\n{answer}"
        return answer

    def rag_with_sources(self, query: str, source_files: list[str] | str | None = None) -> dict:
        """RAG 问答并返回结构化来源信息。

        confidence 为向量相关性置信度（与 confidence_threshold 同源，决定是否加低置信提示）；
        每个来源保留 score（向量相关性）与 rerank_score（CrossEncoder 展示分，未校准）。
        低置信度判断使用完整精度；对外的 confidence 与 sources[0]["score"] 均为四位小数，
        两者恒相等（同一个分值取整而来）。
        """
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
        display_confidence = round(confidence, 4)

        logger.info({
            "event": "rag_with_sources",
            "query": log_safe_text(query),
            "doc_count": len(context_docs),
            "sources": log_safe_value([s["document"] for s in sources]),
            "confidence": display_confidence,
        })

        answer = self._invoke_chain(query, context)
        if confidence < self.confidence_threshold:
            answer = f"{self.LOW_CONFIDENCE_PREFIX}以下回答可能不够准确，建议进一步确认。\n\n{answer}"

        return {
            "answer": answer,
            "sources": sources,
            "confidence": display_confidence,
        }


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
