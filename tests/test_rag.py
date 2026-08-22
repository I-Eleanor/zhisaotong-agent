"""RAG 流程测试：向量检索、MD5 增量去重、CrossEncoder 重排。

全部使用假 Embedding（见 conftest），不加载真实 Embedding 模型。
重排测试加载本地 CrossEncoder 权重（已在沙箱就绪），离线可用。
置信度语义：relevance_score（向量相关性）专用于置信度判断；
rerank_score（CrossEncoder 展示分）只用于排序与展示。
"""

import math

import pytest
from langchain_core.documents import Document

from rag.rag_service import RagSummarizeService
from rag.reranker import CrossEncoderReranker, NoopReranker
from tests.conftest import FakeEmbeddings
from utils.config_handler import chroma_conf


def test_vector_retriever_returns_added_doc(temp_vector_store):
    docs = temp_vector_store.get_retriever(k=1).invoke("滤网")
    assert docs and any("滤网" in d.page_content for d in docs)


def test_md5_incremental_dedup(tmp_path, monkeypatch):
    """升级计划 7.2.2：基于文件 MD5 的增量入库，重复加载不应重复写入。"""
    import rag.vector_store as vmod
    from rag.vector_store import VectorStoreService

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    (data_dir / "sample.txt").write_text("扫地机器人滤网每三个月更换。\n电池避免长期亏电存放。\n", encoding="utf-8")

    md5_store = tmp_path / "md5.txt"

    patched = dict(chroma_conf)
    patched["data_path"] = str(data_dir)
    patched["md5_hex_store"] = str(md5_store)
    patched["chunk_size"] = 50
    patched["chunk_overlap"] = 0
    patched["separators"] = ["\n", "。", ""]
    monkeypatch.setattr(vmod, "chroma_conf", patched)

    vs = VectorStoreService(
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "store"),
        collection_name="dedup_test",
    )
    vs.load_document()
    count1 = vs.vector_store._collection.count()
    vs.load_document()  # 第二次应被 MD5 跳过
    count2 = vs.vector_store._collection.count()

    assert count1 > 0, "首次加载应有分块写入"
    assert count1 == count2, "MD5 增量入库：第二次加载不应重复写入"
    assert md5_store.read_text(encoding="utf-8").strip() != "", "MD5 记录文件应被写入"


def test_cross_encoder_reranker_reranks():
    """重排开启后，CrossEncoder 能按相关性重排文档。"""
    from rag.reranker import CrossEncoderReranker

    model_name = chroma_conf.get("reranker_model")
    reranker = CrossEncoderReranker(model_name=model_name, top_k=2)
    reranker._load_model()
    if reranker._model is None:
        pytest.skip("CrossEncoder 模型不可用，跳过重排测试")
    docs = [
        Document(page_content="扫地机器人滤网每三个月更换一次。"),
        Document(page_content="扫地机器人充不进电时应检查充电座与电池触点。"),
        Document(page_content="无关的随机内容 xyz。"),
    ]
    out = reranker.rerank("充不进电怎么办", docs)
    assert len(out) == 2
    assert "充不进电" in out[0].page_content, "与查询最相关的文档应排在最前"


def test_reranker_enabled_in_config():
    assert chroma_conf.get("reranker_enabled") is True


def test_chinese_file_read_with_utf8(tmp_path, monkeypatch):
    """跨平台编码：以 UTF-8 写入中文文本后，txt_loader 应能正确加载。"""
    import rag.vector_store as vmod
    from rag.vector_store import VectorStoreService

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    (data_dir / "中文测试.txt").write_text("扫地机器人充电异常时请检查电源适配器。\n", encoding="utf-8")

    md5_store = tmp_path / "md5.txt"

    patched = dict(chroma_conf)
    patched["data_path"] = str(data_dir)
    patched["md5_hex_store"] = str(md5_store)
    patched["chunk_size"] = 200
    patched["chunk_overlap"] = 0
    patched["separators"] = ["\n", "。", ""]
    monkeypatch.setattr(vmod, "chroma_conf", patched)

    vs = VectorStoreService(
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "store"),
        collection_name="cn_read_test",
    )
    vs.load_document()
    count = vs.vector_store._collection.count()
    assert count > 0, "中文 UTF-8 文件应能被正确加载入库"


def test_reranker_noop_returns_input():
    from rag.reranker import NoopReranker

    docs = [Document(page_content="a"), Document(page_content="b")]
    noop = NoopReranker()
    out = noop.rerank("query", docs)
    assert out == docs, "NoopReranker 应原样返回输入"


def test_create_reranker_noop():
    from rag.reranker import NoopReranker, create_reranker

    r = create_reranker(enabled=False)
    assert isinstance(r, NoopReranker), "enabled=False 应返回 NoopReranker"


def test_reranker_empty_input():
    from rag.reranker import NoopReranker

    noop = NoopReranker()
    assert noop.rerank("q", []) == [], "空输入应返回空列表"


def test_file_format_whitelist_in_vector_store(tmp_path, monkeypatch):
    import rag.vector_store as vmod
    from rag.vector_store import VectorStoreService

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    (data_dir / "good.txt").write_text("有效内容", encoding="utf-8")
    (data_dir / "bad.exe").write_text("应被忽略", encoding="utf-8")

    md5_store = tmp_path / "md5.txt"
    patched = dict(chroma_conf)
    patched["data_path"] = str(data_dir)
    patched["md5_hex_store"] = str(md5_store)
    patched["chunk_size"] = 200
    patched["chunk_overlap"] = 0
    patched["separators"] = ["\n", "。", ""]
    monkeypatch.setattr(vmod, "chroma_conf", patched)

    vs = VectorStoreService(
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "store"),
        collection_name="whitelist_test",
    )
    vs.load_document()
    assert vs.vector_store._collection.count() > 0, "白名单内文件应被加载"
    md5_lines = md5_store.read_text(encoding="utf-8").strip().splitlines()
    assert len(md5_lines) == 1, "只有 1 个白名单内文件，MD5 记录应为 1 行"


def test_chromadb_temp_dir_write_and_retrieve(tmp_path, monkeypatch):
    """集成测试：ChromaDB 在临时目录写入后可检索"""
    import rag.vector_store as vmod
    from rag.vector_store import VectorStoreService

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("扫地机器人滤网需要定期清洁。\n电池寿命约两到三年。\n", encoding="utf-8")

    md5_store = tmp_path / "md5.txt"
    patched = dict(chroma_conf)
    patched["data_path"] = str(data_dir)
    patched["md5_hex_store"] = str(md5_store)
    patched["chunk_size"] = 100
    patched["chunk_overlap"] = 0
    patched["separators"] = ["\n", "。", ""]
    monkeypatch.setattr(vmod, "chroma_conf", patched)

    persist_dir = str(tmp_path / "store")
    vs = VectorStoreService(
        embedding_function=FakeEmbeddings(),
        persist_directory=persist_dir,
        collection_name="integration_test",
    )
    vs.load_document()
    assert vs.vector_store._collection.count() > 0

    retriever = vs.get_retriever(k=1)
    docs = retriever.invoke("滤网")
    assert docs and any("滤网" in d.page_content for d in docs), "应能检索到滤网相关内容"


# ------------------------------------------------------------- 置信度语义：只用 relevance_score

def _conf_service(docs, threshold=0.3, reranker=None):
    """构造注入固定检索结果的 RAG 服务（不依赖向量库与真实 LLM）。

    retriever_docs 返回给定文档的拷贝；reranker 决定重排行为；
    _invoke_chain 固定返回占位回答，测试只关注置信度语义。
    """
    svc = RagSummarizeService.__new__(RagSummarizeService)
    svc.confidence_threshold = threshold
    svc.retriever_docs = lambda query, source_files=None: [
        Document(page_content=d.page_content, metadata=dict(d.metadata)) for d in docs
    ]
    svc.reranker = reranker or NoopReranker()
    svc._invoke_chain = lambda query, context: "模型回答"
    return svc


def _scored_doc(rel, rerank=None, name="a.txt", chunk=0):
    metadata = {"source_file": name, "chunk_index": chunk, "page": -1, "relevance_score": rel}
    if rerank is not None:
        metadata["rerank_score"] = rerank
    return Document(page_content=f"doc-{name}-{chunk}", metadata=metadata)


def test_confidence_uses_relevance_even_if_rerank_lower():
    """relevance_score=0.8、rerank_score=0.1 → 置信度必须为 0.8。"""
    svc = _conf_service([_scored_doc(0.8, rerank=0.1)])
    assert svc._top_confidence(svc.retriever_docs("q")) == pytest.approx(0.8)
    result = svc.rag_with_sources("q")
    assert result["confidence"] == pytest.approx(0.8)
    assert not result["answer"].startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX)


def test_high_rerank_score_does_not_raise_confidence():
    """relevance_score=0.2、rerank_score=0.99 → 仍判定低置信度（rerank 不参与判断）。"""
    svc = _conf_service([_scored_doc(0.2, rerank=0.99)])
    result = svc.rag_with_sources("q")
    assert result["confidence"] == pytest.approx(0.2)
    assert result["confidence"] < svc.confidence_threshold
    assert result["answer"].startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX), (
        "置信度低于阈值应加低置信提示"
    )


def test_missing_relevance_score_confidence_is_zero():
    """缺失 relevance_score → 置信度 0.0。"""
    svc = _conf_service([Document(page_content="x", metadata={})])
    assert svc._top_confidence(svc.retriever_docs("q")) == 0.0


def test_invalid_relevance_score_returns_zero_without_exception():
    """None / 字符串 / NaN / 正负无穷：不抛异常，一律返回 0.0。"""
    svc = RagSummarizeService.__new__(RagSummarizeService)
    for bad in [None, "not-a-number", float("nan"), float("inf"), float("-inf")]:
        doc = Document(page_content="x", metadata={"relevance_score": bad})
        assert svc._top_confidence([doc]) == 0.0, f"非法值 {bad!r} 应按 0.0 处理"


def test_relevance_score_clamped_to_unit_range():
    """越界分数钳制到 [0, 1]：<0 → 0.0，>1 → 1.0。"""
    svc = RagSummarizeService.__new__(RagSummarizeService)
    low = Document(page_content="x", metadata={"relevance_score": -0.5})
    high = Document(page_content="x", metadata={"relevance_score": 1.7})
    assert svc._top_confidence([low]) == 0.0
    assert svc._top_confidence([high]) == 1.0


class _StubCrossEncoder:
    """固定打分：对输入顺序返回 [0.2, 3.5]，使第二篇文档重排后居首。"""

    def predict(self, pairs):
        return [0.2, 3.5]


def test_confidence_takes_first_doc_relevance_after_rerank():
    """CrossEncoder 重排改变顺序后，置信度取重排后第一篇自身的 relevance_score。"""
    reranker = CrossEncoderReranker(top_k=2)
    reranker._model = _StubCrossEncoder()
    # 召回顺序：A（rel 0.9）在前；Stub 给 B 更高分 → 重排后 B 居首
    svc = _conf_service(
        [_scored_doc(0.9, name="a.txt"), _scored_doc(0.4, name="b.txt", chunk=1)],
        reranker=reranker,
    )
    result = svc.rag_with_sources("q")
    assert result["confidence"] == pytest.approx(0.4), (
        "置信度应为重排后第一篇（b.txt）自身的向量相关性 0.4"
    )
    assert result["sources"][0]["document"] == "b.txt"
    assert result["sources"][0]["score"] == pytest.approx(0.4)
    assert result["sources"][0]["rerank_score"] == pytest.approx(
        1.0 / (1.0 + math.exp(-3.5)), abs=1e-4
    ), "重排分仍写入 sources 供展示，但不影响置信度"


def test_rag_with_sources_confidence_matches_first_source_score():
    """rag_with_sources 的 confidence 与第一来源的 score 一致（同一分数语义）。"""
    svc = _conf_service([_scored_doc(0.66, rerank=0.42), _scored_doc(0.31, name="b.txt", chunk=1)])
    result = svc.rag_with_sources("q")
    assert result["confidence"] == pytest.approx(0.66)
    assert result["confidence"] == pytest.approx(result["sources"][0]["score"])
    assert result["sources"][0]["rerank_score"] == pytest.approx(0.42)
    assert result["sources"][1]["rerank_score"] is None, "无重排分的来源应保持 None"


# ------------------------------------------------------------- 边界修正：布尔与展示分清洗

def test_bool_relevance_score_treated_as_invalid():
    """True / False 不是合法向量分数：均按非法值返回 0.0（bool 是 int 子类，需显式排除）。"""
    svc = RagSummarizeService.__new__(RagSummarizeService)
    for bad in [True, False]:
        doc = Document(page_content="x", metadata={"relevance_score": bad})
        assert svc._top_confidence([doc]) == 0.0, f"布尔值 {bad} 不应被视为合法分数"


def test_parse_optional_display_score_invalid_values_return_none():
    """展示分解析：bool / 字符串 / None / NaN / 正负无穷 → None（JSON 可安全序列化）。"""
    from rag.rag_service import _parse_optional_display_score

    for bad in [True, False, "x", None, float("nan"), float("inf"), float("-inf")]:
        assert _parse_optional_display_score(bad) is None, f"非法值 {bad!r} 应返回 None"


def test_parse_optional_display_score_clamps_and_rounds():
    """展示分解析：合法数字钳制到 [0,1] 并保留四位小数。"""
    from rag.rag_service import _parse_optional_display_score

    assert _parse_optional_display_score(-0.2) == 0.0
    assert _parse_optional_display_score(1.2) == 1.0
    assert _parse_optional_display_score(0.123456) == pytest.approx(0.1235)
    assert _parse_optional_display_score(0.5) == 0.5


def test_extract_sources_rerank_score_never_leaks_invalid_numbers():
    """非法 rerank_score 经 _extract_sources 清洗后为 None，不会输出 NaN / Infinity。"""
    svc = RagSummarizeService.__new__(RagSummarizeService)
    docs = [
        Document(page_content="x", metadata={"source_file": "a.txt", "chunk_index": 0,
                                             "relevance_score": 0.5, "rerank_score": True}),
        Document(page_content="y", metadata={"source_file": "b.txt", "chunk_index": 1,
                                             "relevance_score": 0.5, "rerank_score": float("nan")}),
    ]
    sources = svc._extract_sources(docs)
    assert sources[0]["rerank_score"] is None, "布尔 rerank_score 应清洗为 None"
    assert sources[1]["rerank_score"] is None, "NaN rerank_score 应清洗为 None"
    import json
    serialized = json.dumps(sources, allow_nan=False)  # 残留 NaN/Infinity 会抛 ValueError
    assert "NaN" not in serialized and "Infinity" not in serialized, "输出必须是严格合法 JSON"


def test_confidence_and_source_score_rounded_consistently():
    """relevance_score=0.666666 → confidence == sources[0]["score"] == 0.6667（统一四位小数）。"""
    svc = _conf_service([_scored_doc(0.666666)])
    result = svc.rag_with_sources("q")
    assert result["confidence"] == 0.6667
    assert result["sources"][0]["score"] == 0.6667
    assert result["confidence"] == result["sources"][0]["score"], (
        "顶层 confidence 与第一来源 score 必须是同一个规范化值，不得一个全精度一个四舍五入"
    )


# ------------------------------------------------------------- 内部精度与展示精度分离

def test_internal_full_precision_below_threshold_display_rounded():
    """relevance_score=0.29996、阈值 0.3：全精度判低置信度，展示值四舍五入为 0.3。

    修复前 _top_confidence 先 round 到 0.3，0.3 < 0.3 不成立 → 边界被展示精度翻转。
    """
    svc = _conf_service([_scored_doc(0.29996)])
    assert svc.confidence_threshold == 0.3

    assert svc._top_confidence(svc.retriever_docs("q")) == pytest.approx(0.29996), (
        "内部置信度必须保留完整精度，不取整"
    )

    result = svc.rag_with_sources("q")
    assert result["answer"].startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX), (
        "全精度 0.29996 < 0.3，必须判定低置信度（不得被展示精度干扰）"
    )
    assert result["confidence"] == 0.3
    assert result["sources"][0]["score"] == 0.3
    assert result["confidence"] == result["sources"][0]["score"]

    low = svc.rag_summarize("q")
    assert low.startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX), (
        "rag_summarize 同样必须用完整精度判断"
    )


def test_internal_full_precision_above_threshold_display_rounded():
    """relevance_score=0.30004、阈值 0.3：全精度不判低置信度，展示值同为 0.3。

    与上一用例合看：展示值完全相同（0.3），但业务判断结果相反——
    证明内部判断未被展示精度影响。
    """
    svc = _conf_service([_scored_doc(0.30004)])

    assert svc._top_confidence(svc.retriever_docs("q")) == pytest.approx(0.30004)

    result = svc.rag_with_sources("q")
    assert not result["answer"].startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX), (
        "全精度 0.30004 >= 0.3，不得判定低置信度"
    )
    assert result["confidence"] == 0.3

    high = svc.rag_summarize("q")
    assert not high.startswith(RagSummarizeService.LOW_CONFIDENCE_PREFIX)


# ------------------------------------------------------------- CrossEncoder 非有限分数

class _NonFiniteCrossEncoder:
    """固定打分：[NaN, 3.5, -inf, 1.0, None]，混合非法值与有限值。"""

    def predict(self, pairs):
        return [float("nan"), 3.5, float("-inf"), 1.0, None]


def test_reranker_non_finite_scores_sorted_after_finite():
    """NaN / ±inf / 非数值分数不参与正常排序：有限分数降序在前，非法值排最后。"""
    reranker = CrossEncoderReranker(top_k=5)
    reranker._model = _NonFiniteCrossEncoder()
    docs = [
        Document(page_content="A", metadata={"relevance_score": 0.1}),
        Document(page_content="B", metadata={"relevance_score": 0.2}),
        Document(page_content="C", metadata={"relevance_score": 0.3}),
        Document(page_content="D", metadata={"relevance_score": 0.4}),
        Document(page_content="E", metadata={"relevance_score": 0.5}),
    ]
    out = reranker.rerank("q", docs)

    assert [d.page_content for d in out] == ["B", "D", "A", "C", "E"], (
        "有限分数降序在前（B=3.5 > D=1.0），非有限分数按原始相对顺序排在末尾"
    )

    score_by_doc = {d.page_content: d.metadata.get("rerank_score") for d in out}
    assert score_by_doc["B"] == pytest.approx(1.0 / (1.0 + math.exp(-3.5)), abs=1e-4)
    assert score_by_doc["D"] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), abs=1e-4)
    for name in ("A", "C", "E"):
        assert "rerank_score" not in next(d for d in out if d.page_content == name).metadata, (
            f"非有限分数文档 {name} 不得写入 rerank_score"
        )


def test_reranker_all_scores_non_finite_returns_original_order():
    """全部分数非法：按原始顺序返回前 top_k，不抛异常，不写入任何 rerank_score。"""
    class _AllInvalidCrossEncoder:
        def predict(self, pairs):
            return [float("nan"), float("inf"), float("-inf")]

    reranker = CrossEncoderReranker(top_k=2)
    reranker._model = _AllInvalidCrossEncoder()
    docs = [
        Document(page_content="A", metadata={"relevance_score": 0.9}),
        Document(page_content="B", metadata={"relevance_score": 0.5}),
        Document(page_content="C", metadata={"relevance_score": 0.3}),
    ]
    out = reranker.rerank("q", docs)

    assert [d.page_content for d in out] == ["A", "B"], "全非法时按原始顺序取前 top_k"
    for d in out:
        assert "rerank_score" not in d.metadata, "非法分数不得写入 rerank_score"


# ----------------------------------------------------------------- 异常日志收敛（P1-6）
def test_load_document_failure_log_is_structured_and_safe(tmp_path, monkeypatch, caplog):
    """文件加载失败：结构化日志只含文件名（无绝对路径 / traceback / exc_info / 密钥），降级继续。"""
    import logging

    import rag.vector_store as vmod
    from rag.vector_store import VectorStoreService

    data_dir = tmp_path / "kb"
    data_dir.mkdir()
    (data_dir / "损坏文档.txt").write_text("内容", encoding="utf-8")
    md5_store = tmp_path / "md5.txt"

    patched = dict(chroma_conf)
    patched["data_path"] = str(data_dir)
    patched["md5_hex_store"] = str(md5_store)
    patched["chunk_size"] = 50
    patched["chunk_overlap"] = 0
    patched["separators"] = ["\n", "。", ""]
    monkeypatch.setattr(vmod, "chroma_conf", patched)

    def broken_loader(path):
        raise RuntimeError(f"解析失败：{path} api_key=sk-VS-998877665544")

    monkeypatch.setattr(vmod, "txt_loader", broken_loader)

    vs = VectorStoreService(
        embedding_function=FakeEmbeddings(),
        persist_directory=str(tmp_path / "store"),
        collection_name="load_fail_log_test",
    )
    with caplog.at_level(logging.ERROR, logger="agent"):
        vs.load_document()  # 不得向上抛异常（降级跳过该文件）

    entries = [
        r.msg for r in caplog.records
        if isinstance(r.msg, dict) and r.msg.get("event") == "knowledge_load_failed"
    ]
    assert entries, "应记录结构化失败日志"
    entry = entries[-1]
    assert entry["file"] == "损坏文档.txt", "只记录文件名"
    assert str(tmp_path) not in str(entry), "不得记录本地绝对路径"
    assert "sk-VS-998877665544" not in entry["error_msg"], "原始密钥不得进日志"
    assert entry["error_type"] == "RuntimeError"
    assert "traceback" not in entry, "不得记录 traceback 字段"

    record = next(r for r in caplog.records
                  if isinstance(r.msg, dict) and r.msg.get("event") == "knowledge_load_failed")
    assert record.exc_info is None, "禁止 exc_info=True"

    assert vs.count() == 0, "失败文件不应入库（降级行为不变）"
