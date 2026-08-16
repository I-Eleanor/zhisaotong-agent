"""工具类测试：file_handler 修复、config_handler、path_tool、logger、模型工厂、prompt 加载。"""
import os

from utils import file_handler, path_tool
from utils.config_handler import agent_conf, chroma_conf, prompts_conf, rag_conf


def test_listdir_returns_empty_for_nondir():
    result = file_handler.listdir_with_allowed_type("/不存在的路径/xyz", ("txt", "pdf"))
    assert result == [], f"非目录应返回 []，实际返回 {result!r}"


def test_listdir_filters_by_extension(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "b.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "c.md").write_text("x", encoding="utf-8")
    found = file_handler.listdir_with_allowed_type(str(tmp_path), ("txt", "pdf"))
    names = {os.path.basename(p) for p in found}
    assert names == {"a.txt", "b.pdf"}


def test_md5_deterministic_and_distinct(tmp_path):
    f1 = tmp_path / "f1.txt"
    f2 = tmp_path / "f2.txt"
    f1.write_text("hello", encoding="utf-8")
    f2.write_text("world", encoding="utf-8")
    h1 = file_handler.get_file_md5_hex(str(f1))
    h2 = file_handler.get_file_md5_hex(str(f2))
    assert h1 == file_handler.get_file_md5_hex(str(f1)), "同一文件 MD5 必须一致"
    assert h1 != h2, "不同文件 MD5 必须不同"
    assert isinstance(h1, str) and len(h1) == 32


def test_md5_missing_file_returns_none(tmp_path):
    assert file_handler.get_file_md5_hex(str(tmp_path / "nope.txt")) is None


def test_config_handler_loads_all():
    assert isinstance(rag_conf, dict) and rag_conf.get("chat_model_name")
    assert isinstance(chroma_conf, dict) and chroma_conf.get("collection_name")
    assert isinstance(agent_conf, dict)
    assert isinstance(prompts_conf, dict)


def test_path_tool_resolves_relative():
    abs_path = path_tool.get_abs_path("config/rag.yml")
    assert os.path.isabs(abs_path)
    assert abs_path.endswith("config/rag.yml") or abs_path.endswith("config\\rag.yml")


def test_logger_handler_emits():
    from utils.logger_handler import logger
    logger.info({"event": "test_log", "value": 1})


# ----------------------------------------------------------------- 模型工厂懒加载与缓存
def test_model_factory_lazy_and_cache():
    import model.factory as mf
    mf.reset_models()
    assert mf._chat_model is None, "初始应为 None（懒加载）"
    assert mf._embed_model is None


def test_model_factory_reset():
    import model.factory as mf
    mf.reset_models()
    assert mf._chat_model is None
    assert mf._embed_model is None


# ----------------------------------------------------------------- Prompt 缺失行为
def test_prompt_missing_key_raises():
    from utils.prompt_loader import _cache, _load_prompt
    _cache.pop("__nonexistent_key__", None)
    import pytest
    with pytest.raises(KeyError):
        _load_prompt("__nonexistent_key__", "测试")


# ----------------------------------------------------------------- 文件格式白名单
def test_file_whitelist_rejects_exe(tmp_path):
    (tmp_path / "good.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "bad.exe").write_text("nope", encoding="utf-8")
    found = file_handler.listdir_with_allowed_type(str(tmp_path), ("txt", "pdf"))
    names = {os.path.basename(p) for p in found}
    assert "good.txt" in names
    assert "bad.exe" not in names
