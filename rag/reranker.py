import math

from langchain_core.documents import Document

from utils.logger_handler import log_safe_text, logger, safe_exception_fields


def _sigmoid(x: float) -> float:
    """把 CrossEncoder 的原始 logit 分数压到 (0, 1)，保持单调性。

    仅用于展示归一化：不是经过校准的概率，不参与置信度判断
    （是否拒答由向量检索的 relevance_score 决定）。
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _to_float(raw) -> float:
    """把模型原始分数转为 float；无法转换（None / 字符串等）统一按 NaN 处理。"""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


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
            logger.error({"event": "reranker_load_error", "model": self.model_name, "stage": "model", **safe_exception_fields(e)})
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

            scored_docs = [
                (_to_float(raw), doc) for raw, doc in zip(scores, documents, strict=True)
            ]
            finite_docs = [(s, d) for s, d in scored_docs if math.isfinite(s)]
            non_finite_docs = [(s, d) for s, d in scored_docs if not math.isfinite(s)]

            # 非有限分数（NaN / ±inf / 非数值）不参与正常排序：
            # 有限分数降序在前，非有限分数整体排在末尾并保持原始相对顺序；
            # 全部非法时即原始顺序，取前 top_k，不抛异常。
            finite_docs.sort(key=lambda x: x[0], reverse=True)
            ordered_docs = finite_docs + non_finite_docs
            top_docs = [doc for _, doc in ordered_docs[:self.top_k]]

            # CrossEncoder 分数单独写入 rerank_score（sigmoid 归一化到 0~1）：
            # 只用于排序与前端展示，不覆盖向量检索写入的 relevance_score，
            # 也不参与上层置信度判断（是否拒答由 relevance_score 决定）。
            # 非有限分数不写入 rerank_score，避免非法值进入 metadata 与 API 输出。
            for raw_score, doc in ordered_docs[:self.top_k]:
                if math.isfinite(raw_score):
                    doc.metadata["rerank_score"] = round(_sigmoid(raw_score), 4)
                else:
                    doc.metadata.pop("rerank_score", None)

            if non_finite_docs:
                logger.warning({
                    "event": "reranker_non_finite_scores",
                    "query": log_safe_text(query),
                    "input_count": len(documents),
                    "non_finite_count": len(non_finite_docs),
                })

            logger.info({
                "event": "reranker_success",
                "query": log_safe_text(query),
                "input_count": len(documents),
                "output_count": len(top_docs),
                "top_scores": [s for s, _ in finite_docs[:self.top_k]],
            })

            return top_docs
        except Exception as e:
            logger.error({"event": "reranker_error", "stage": "retrieval", **safe_exception_fields(e)})
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
