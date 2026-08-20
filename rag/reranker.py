import math

from langchain_core.documents import Document

from utils.logger_handler import logger


def _sigmoid(x: float) -> float:
    """把 CrossEncoder 的原始 logit 分数压到 (0, 1)，保持单调性。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 3):
        self.model_name = model_name
        self.top_k = top_k
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info({"event": "reranker_loaded", "model": self.model_name})
        except Exception as e:
            logger.error({"event": "reranker_load_error", "model": self.model_name, "error": str(e), "stage": "model"})
            self._model = None

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return documents

        self._load_model()

        if self._model is None:
            logger.warning({"event": "reranker_fallback", "reason": "模型加载失败，返回原始结果"})
            return documents[:self.top_k]

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self._model.predict(pairs)

            scored_docs = list(zip(scores, documents, strict=True))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            top_docs = [doc for _, doc in scored_docs[:self.top_k]]

            # CrossEncoder 分数单独写入 rerank_score（sigmoid 归一化到 0~1），
            # 不覆盖向量检索写入的 relevance_score，两者语义独立。
            for raw_score, doc in scored_docs[:self.top_k]:
                try:
                    doc.metadata["rerank_score"] = round(_sigmoid(float(raw_score)), 4)
                except (TypeError, ValueError):
                    doc.metadata.pop("rerank_score", None)

            logger.info({
                "event": "reranker_success",
                "query": query,
                "input_count": len(documents),
                "output_count": len(top_docs),
                "top_scores": [float(s) for s, _ in scored_docs[:self.top_k]],
            })

            return top_docs
        except Exception as e:
            logger.error({"event": "reranker_error", "error": str(e), "stage": "retrieval"})
            return documents[:self.top_k]


class NoopReranker:
    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        return documents


Reranker = CrossEncoderReranker | NoopReranker


def create_reranker(enabled: bool = False, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 3) -> Reranker:
    if not enabled:
        logger.info({"event": "reranker_init", "mode": "noop"})
        return NoopReranker()
    logger.info({"event": "reranker_init", "mode": "cross_encoder", "model": model_name, "top_k": top_k})
    return CrossEncoderReranker(model_name=model_name, top_k=top_k)
