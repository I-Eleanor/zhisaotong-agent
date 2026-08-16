"""配置验证器测试：覆盖 validate_before_use / validate_env_vars / validate_paths 的各分支。"""

import pytest

from utils.config_validator import ConfigValidationError, validate_before_use, validate_env_vars, validate_paths


def test_validate_chat_model_missing_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigValidationError, match="DEEPSEEK_API_KEY"):
        validate_before_use("chat_model")


def test_validate_chat_model_with_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    validate_before_use("chat_model")


def test_validate_embedding_local_path_not_exists(monkeypatch):
    from utils.config_handler import rag_conf
    monkeypatch.setattr("utils.config_validator.rag_conf", {**rag_conf, "embedding_model_name": "local-embedding", "embedding_local_path": "/nonexistent/path"})
    with pytest.raises(ConfigValidationError, match="本地 Embedding"):
        validate_before_use("embedding")


def test_validate_embedding_dashscope_missing_key(monkeypatch):
    from utils.config_handler import rag_conf
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr("utils.config_validator.rag_conf", {**rag_conf, "embedding_model_name": "dashscope-embedding"})
    with pytest.raises(ConfigValidationError, match="DASHSCOPE_API_KEY"):
        validate_before_use("embedding")


def test_validate_env_vars_missing_deepseek(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    errors = validate_env_vars()
    assert any("DEEPSEEK_API_KEY" in e for e in errors)


def test_validate_env_vars_with_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    errors = validate_env_vars()
    deepseek_errors = [e for e in errors if "DEEPSEEK" in e[0]]
    assert len(deepseek_errors) == 0


def test_validate_paths_returns_list():
    errors = validate_paths()
    assert isinstance(errors, list)
