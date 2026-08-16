"""RAG 流程测试：向量检索、MD5 增量去重、CrossEncoder 重排。

全部使用假 Embedding（见 conftest），不加载真实 Embedding 模型。
重排测试加载本地 CrossEncoder 权重（已在沙箱就绪），离线可用。
"""

from langchain_core.documents import Document

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
