"""P0-1 RAG 置信度修复测试。

覆盖指令要求的场景：
- 无检索结果 / 高置信度 / 低置信度 / 分数缺失
- source_files 过滤
- 重排前后仍保留 relevance_score
- rerank_score 与 relevance_score 不混淆
"""
import math

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.rag_service import RagSummarizeService
from rag.reranker import CrossEncoderReranker, NoopReranker
from rag.vector_store import VectorStoreService, _normalize_relevance_score


class FixedEmbeddings:
    """可控假 Embedding：按关键词映射到正交单位向量，余弦相似度可精确预测。"""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self._keys = ("滤网", "电池", "充电", "噪音")

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for i, key in enumerate(self._keys):
            if key in text and i < self.dim:
                vec[i] = 1.0
                return vec
        vec[self.dim - 1] = 1.0
        return vec

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _make_service(tmp_path, monkeypatch, docs=None, collection="conf_test"):
    """构造使用 NoopReranker 的确定性 RAG 服务（不依赖 CrossEncoder 权重与真实 LLM）。"""
    monkeypatch.setattr("rag.rag_service.create_reranker", lambda **kwargs: NoopReranker())
    vs = VectorStoreService(
        embedding_function=FixedEmbeddings(),
        persist_directory=str(tmp_path / collection),
        collection_name=collection,
    )
    if docs:
        vs.vector_store.add_documents(docs)
    return RagSummarizeService(vector_store=vs, model=FakeListChatModel(responses=["这是测试回答。"]))


def _doc(text: str, source: str, chunk: int = 0) -> Document:
    return Document(page_content=text, metadata={"source_file": source, "chunk_index": chunk, "page": -1})


@pytest.fixture
def rag_service(tmp_path, monkeypatch):
    return _make_service(
        tmp_path,
        monkeypatch,
        docs=[
            _doc("滤网更换方法", "a.txt"),
            _doc("电池保养技巧", "b.txt"),
        ],
    )


@pytest.fixture
def empty_service(tmp_path, monkeypatch):
    return _make_service(tmp_path, monkeypatch, docs=None, collection="empty_test")


# ------------------------------------------------------------- search_with_scores

def test_search_with_scores_writes_relevance_in_unit_range(rag_service):
    docs = rag_service.retriever_docs("滤网")
    assert docs, "应检索到文档"
    for d in docs:
        assert 0.0 <= d.metadata["relevance_score"] <= 1.0
    assert "滤网" in docs[0].page_content
    assert docs[0].metadata["relevance_score"] == pytest.approx(1.0), "查询与文档同关键词 → 余弦相似度 1.0"


def test_normalize_relevance_score_edges():
    assert _normalize_relevance_score(None) is None
    assert _normalize_relevance_score("bad") is None
    assert _normalize_relevance_score(float("nan")) is None
    assert _normalize_relevance_score(float("inf")) is None
    assert _normalize_relevance_score(-15.5) == 0.0
    assert _normalize_relevance_score(2.0) == 1.0
    assert _normalize_relevance_score(0.7) == pytest.approx(0.7)


def test_source_files_filter(rag_service):
    # 滤网文档在 a.txt；限定 b.txt 后检索范围只剩电池文档（向量检索不做阈值过滤，
    # 仍会返回范围内最近邻，但其相关性分数应为 0 → 上层按低置信度处理）
    docs = rag_service.retriever_docs("滤网", source_files=["b.txt"])
    assert all(d.metadata["source_file"] == "b.txt" for d in docs)
    assert docs and all(d.metadata["relevance_score"] == pytest.approx(0.0) for d in docs)

    docs2 = rag_service.retriever_docs("电池", source_files=["b.txt"])
    assert len(docs2) == 1
    assert docs2[0].metadata["source_file"] == "b.txt"
    assert docs2[0].metadata["relevance_score"] == pytest.approx(1.0), "过滤后分数仍应写入"


# ------------------------------------------------------------- 无检索结果

def test_no_results_all_entries_consistent(empty_service):
    assert empty_service.build_context("任意问题") == ""
    assert empty_service.rag_summarize("任意问题") == RagSummarizeService.NO_RESULT_MESSAGE

    result = empty_service.rag_with_sources("任意问题")
    assert result["sources"] == []
    assert result["confidence"] == 0.0
    assert result["answer"] == RagSummarizeService.NO_RESULT_MESSAGE


# ------------------------------------------------------------- 高 / 低置信度

def test_high_confidence_no_warning_prefix(rag_service):
    result = rag_service.rag_with_sources("滤网")
    assert result["confidence"] == pytest.approx(1.0)
    assert result["confidence"] >= rag_service.confidence_threshold
    assert not result["answer"].startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX)
    assert result["sources"], "应返回结构化来源"


def test_low_confidence_adds_warning_prefix(rag_service):
    result = rag_service.rag_with_sources("充电")
    assert result["confidence"] < rag_service.confidence_threshold
    assert result["answer"].startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX)


def test_rag_summarize_shares_confidence_chain(rag_service):
    """rag_summarize 与 rag_with_sources 共用同一条置信度链路（此前 summarize 不做置信度检查）。"""
    low = rag_service.rag_summarize("充电")
    assert low.startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX)

    high = rag_service.rag_summarize("滤网")
    assert not high.startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX)


def test_confidence_no_longer_always_zero(rag_service):
    """回归：修复前 rag_with_sources 的 confidence 恒为 0.0（分数从未写入）。"""
    result = rag_service.rag_with_sources("滤网")
    assert result["confidence"] > 0.0


# ------------------------------------------------------------- 分数缺失

def test_missing_score_treated_as_low_confidence():
    svc = RagSummarizeService.__new__(RagSummarizeService)
    doc_without_score = Document(page_content="x", metadata={})
    assert svc._top_confidence([]) == 0.0
    assert svc._top_confidence([doc_without_score]) == 0.0, "分数缺失必须按低置信度处理"


def test_top_confidence_prefers_rerank_score():
    svc = RagSummarizeService.__new__(RagSummarizeService)
    only_vector = Document(page_content="x", metadata={"relevance_score": 0.2})
    both_scores = Document(page_content="x", metadata={"relevance_score": 0.2, "rerank_score": 0.9})
    assert svc._top_confidence([only_vector]) == pytest.approx(0.2)
    assert svc._top_confidence([both_scores]) == pytest.approx(0.9)


# ------------------------------------------------------------- 重排分数分离

class _StubCrossEncoder:
    def predict(self, pairs):
        return [0.2, 3.5]


def test_reranker_writes_rerank_score_and_preserves_relevance():
    reranker = CrossEncoderReranker(top_k=2)
    reranker._model = _StubCrossEncoder()

    docs = [
        Document(page_content="文档A", metadata={"relevance_score": 0.9}),
        Document(page_content="文档B", metadata={"relevance_score": 0.4}),
    ]
    out = reranker.rerank("查询", docs)

    top = out[0]
    assert top.page_content == "文档B", "CrossEncoder 分数最高的文档应排第一"
    assert top.metadata["relevance_score"] == pytest.approx(0.4), "重排不得覆盖向量相关性分数"
    assert top.metadata["rerank_score"] == pytest.approx(1.0 / (1.0 + math.exp(-3.5)), abs=1e-4), "rerank_score 应为 sigmoid(raw)"
    assert top.metadata["rerank_score"] != top.metadata["relevance_score"], "两个分数语义不同，不得混淆"


def test_noop_reranker_keeps_relevance_without_rerank_score():
    noop = NoopReranker()
    docs = [Document(page_content="x", metadata={"relevance_score": 0.8})]
    out = noop.rerank("q", docs)
    assert out[0].metadata["relevance_score"] == pytest.approx(0.8)
    assert "rerank_score" not in out[0].metadata, "未启用重排时不应出现 rerank_score"


def test_extract_sources_keeps_both_scores():
    svc = RagSummarizeService.__new__(RagSummarizeService)
    docs = [
        Document(page_content="x", metadata={"source_file": "a.txt", "chunk_index": 0,
                                             "relevance_score": 0.7, "rerank_score": 0.95}),
        Document(page_content="y", metadata={"source_file": "b.txt", "chunk_index": 1,
                                             "relevance_score": 0.3}),
    ]
    sources = svc._extract_sources(docs)
    assert sources[0]["score"] == pytest.approx(0.7)
    assert sources[0]["rerank_score"] == pytest.approx(0.95)
    assert sources[1]["score"] == pytest.approx(0.3)
    assert sources[1]["rerank_score"] is None
