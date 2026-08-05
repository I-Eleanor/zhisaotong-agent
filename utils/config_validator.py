import os
from pathlib import Path
from typing import List, Tuple
from utils.path_tool import get_abs_path
from utils.config_handler import rag_conf, prompts_conf, agent_conf


class ConfigValidationError(Exception):
    pass


def validate_env_vars() -> List[Tuple[str, str]]:
    errors = []
    kimi_api_key = os.getenv("KIMI_API_KEY")
    if not kimi_api_key or kimi_api_key == "your_kimi_api_key_here":
        errors.append(("KIMI_API_KEY", "未设置或使用示例值"))

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if rag_conf.get("embedding_model_name") == "dashscope-embedding":
        if not dashscope_api_key or dashscope_api_key == "your_dashscope_api_key_here":
            errors.append(("DASHSCOPE_API_KEY", "未设置或使用示例值"))

    return errors


def validate_paths() -> List[Tuple[str, str]]:
    errors = []

    prompt_paths = [
        ("main_prompt_path", prompts_conf.get("main_prompt_path")),
        ("rag_summarize_prompt_path", prompts_conf.get("rag_summarize_prompt_path")),
        ("report_prompt_path", prompts_conf.get("report_prompt_path")),
    ]

    for name, rel_path in prompt_paths:
        if not rel_path:
            errors.append((name, "配置项缺失"))
            continue
        abs_path = get_abs_path(rel_path)
        if not os.path.exists(abs_path):
            errors.append((name, f"文件不存在: {abs_path}"))

    external_data_path = agent_conf.get("external_data_path")
    if external_data_path:
        abs_path = get_abs_path(external_data_path)
        if not os.path.exists(abs_path):
            errors.append(("external_data_path", f"文件不存在: {abs_path}"))

    embedding_local_path = rag_conf.get("embedding_local_path")
    if embedding_local_path and rag_conf.get("embedding_model_name") == "local-embedding":
        if not os.path.exists(embedding_local_path):
            errors.append(("embedding_local_path", f"目录不存在: {embedding_local_path}"))

    return errors


def validate_model_config() -> List[Tuple[str, str]]:
    errors = []

    chat_model_name = rag_conf.get("chat_model_name")
    if not chat_model_name:
        errors.append(("chat_model_name", "配置项缺失"))

    kimi_base_url = rag_conf.get("kimi_base_url")
    if not kimi_base_url:
        errors.append(("kimi_base_url", "配置项缺失"))

    embedding_model_name = rag_conf.get("embedding_model_name")
    if not embedding_model_name:
        errors.append(("embedding_model_name", "配置项缺失"))

    return errors


def validate_startup():
    all_errors = []

    env_errors = validate_env_vars()
    all_errors.extend([("环境变量", name, msg) for name, msg in env_errors])

    path_errors = validate_paths()
    all_errors.extend([("路径", name, msg) for name, msg in path_errors])

    model_errors = validate_model_config()
    all_errors.extend([("模型配置", name, msg) for name, msg in model_errors])

    if all_errors:
        error_msg = "\n配置校验失败:\n"
        for category, name, msg in all_errors:
            error_msg += f"  [{category}] {name}: {msg}\n"
        error_msg += "\n请检查 .env 文件和 config/*.yml 配置文件"
        raise ConfigValidationError(error_msg)


def validate_before_use(config_type: str):
    if config_type == "chat_model":
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key or api_key == "your_kimi_api_key_here":
            raise ConfigValidationError("KIMI_API_KEY 未正确配置，无法使用聊天模型")

    elif config_type == "embedding":
        model_name = rag_conf.get("embedding_model_name")
        if model_name == "dashscope-embedding":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key or api_key == "your_dashscope_api_key_here":
                raise ConfigValidationError("DASHSCOPE_API_KEY 未正确配置，无法使用在线 Embedding")
        elif model_name == "local-embedding":
            local_path = rag_conf.get("embedding_local_path")
            if not local_path or not os.path.exists(local_path):
                raise ConfigValidationError(f"本地 Embedding 模型路径未正确配置: {local_path}")

    elif config_type == "external_data":
        external_data_path = agent_conf.get("external_data_path")
        if not external_data_path:
            raise ConfigValidationError("external_data_path 配置项缺失")
        abs_path = get_abs_path(external_data_path)
        if not os.path.exists(abs_path):
            raise ConfigValidationError(f"外部数据文件不存在: {abs_path}")