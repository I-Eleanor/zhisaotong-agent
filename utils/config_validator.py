import os
import re

from utils.config_handler import agent_conf, prompts_conf, rag_conf
from utils.path_tool import get_abs_path


class ConfigValidationError(Exception):
    pass


# 支持的 embedding 模式（model/factory.py 的分支依据）
_SUPPORTED_EMBEDDING_MODES = ("local-embedding", "dashscope-embedding")

# 合法 CORS 来源：通配符 * 由应用层警告提示，显式来源须为 http(s)://host[:port]
_VALID_ORIGIN_PATTERN = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d{1,5})?/?$")


def _is_hf_model_id(path: str) -> bool:
    """判断路径是否为 HuggingFace 模型 ID（而非本地路径）。

    HuggingFace 模型 ID 格式为 ``org/model-name``（如 ``sentence-transformers/xxx``），
    既不是 Windows 绝对路径（``D:\\...``）也不是 Unix 绝对路径（``/...``）。
    """
    if not path:
        return False
    if len(path) >= 2 and path[1] == ":":       # Windows 盘符
        return False
    if path.startswith("/"):                     # Unix 绝对路径
        return False
    return "/" in path


def _basename(value: str) -> str:
    """取文件名用于错误提示：不泄漏完整本地路径（P1-13 安全边界）。"""
    return os.path.basename(value)


def validate_env_vars() -> list[tuple[str, str]]:
    errors = []
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key or deepseek_api_key == "your_deepseek_api_key_here":
        errors.append(("DEEPSEEK_API_KEY", "未设置或使用示例值"))

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if rag_conf.get("embedding_model_name") == "dashscope-embedding" and (
        not dashscope_api_key or dashscope_api_key == "your_dashscope_api_key_here"
    ):
        errors.append(("DASHSCOPE_API_KEY", "未设置或使用示例值"))

    return errors


def validate_paths() -> list[tuple[str, str]]:
    """路径存在性校验。

    错误信息只含配置声明的相对路径或文件名（basename），
    不泄漏完整本地绝对路径（P1-13 安全边界）。
    """
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
            errors.append((name, f"文件不存在: {_basename(rel_path)}"))

    external_data_path = agent_conf.get("external_data_path")
    if external_data_path:
        abs_path = get_abs_path(external_data_path)
        if not os.path.exists(abs_path):
            errors.append(("external_data_path", f"文件不存在: {_basename(external_data_path)}"))

    embedding_local_path = rag_conf.get("embedding_local_path")
    if (
        embedding_local_path
        and rag_conf.get("embedding_model_name") == "local-embedding"
        and not os.path.exists(embedding_local_path)
        and not _is_hf_model_id(embedding_local_path)
    ):
        errors.append(("embedding_local_path", f"目录不存在: {_basename(embedding_local_path)}"))

    return errors


def validate_model_config() -> list[tuple[str, str]]:
    """模型相关配置校验：模型名、基址、超时参数。

    超时非法形态：负数 / 零 / 非数值 / 布尔（缺省 60 秒合法）。
    错误信息不含配置原值（原值可能携带任意文本）。
    """
    errors = []

    chat_model_name = rag_conf.get("chat_model_name")
    if not chat_model_name:
        errors.append(("chat_model_name", "配置项缺失"))
    elif not isinstance(chat_model_name, str) or not chat_model_name.strip():
        errors.append(("chat_model_name", "模型名不能为空"))
    elif any(c.isspace() for c in chat_model_name) or "/" in chat_model_name or "\\" in chat_model_name:
        errors.append(("chat_model_name", "模型名含非法字符（空白或路径分隔符）"))

    deepseek_base_url = rag_conf.get("deepseek_base_url")
    if not deepseek_base_url:
        errors.append(("deepseek_base_url", "配置项缺失"))

    embedding_model_name = rag_conf.get("embedding_model_name")
    if not embedding_model_name:
        errors.append(("embedding_model_name", "配置项缺失"))
    elif embedding_model_name not in _SUPPORTED_EMBEDDING_MODES:
        errors.append((
            "embedding_model_name",
            f"不支持的模型类型（支持 {' / '.join(_SUPPORTED_EMBEDDING_MODES)}）",
        ))

    timeout_seconds = rag_conf.get("llm_timeout_seconds", 60)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or timeout_seconds <= 0
    ):
        errors.append(("llm_timeout_seconds", "必须为正数（秒）"))

    return errors


def validate_cors_origins() -> list[tuple[str, str]]:
    """CORS_ORIGINS 格式校验。

    合法形态：通配符 ``*``（应用层另记安全警告）或逗号分隔的
    ``http(s)://host[:port]`` 显式来源列表。空值合法（等价于禁止跨域）。
    """
    raw = os.getenv("CORS_ORIGINS", "*")
    errors: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or item == "*":
            continue
        if not _VALID_ORIGIN_PATTERN.match(item):
            errors.append(("CORS_ORIGINS", f"非法来源格式: {item[:60]}"))
    return errors


def validate_startup():
    all_errors = []

    env_errors = validate_env_vars()
    all_errors.extend([("环境变量", name, msg) for name, msg in env_errors])

    path_errors = validate_paths()
    all_errors.extend([("路径", name, msg) for name, msg in path_errors])

    model_errors = validate_model_config()
    all_errors.extend([("模型配置", name, msg) for name, msg in model_errors])

    cors_errors = validate_cors_origins()
    all_errors.extend([("CORS", name, msg) for name, msg in cors_errors])

    if all_errors:
        error_msg = "\n配置校验失败:\n"
        for category, name, msg in all_errors:
            error_msg += f"  [{category}] {name}: {msg}\n"
        error_msg += "\n请检查 .env 文件和 config/*.yml 配置文件"
        raise ConfigValidationError(error_msg)


def validate_before_use(config_type: str):
    if config_type == "chat_model":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key or api_key == "your_deepseek_api_key_here":
            raise ConfigValidationError("DEEPSEEK_API_KEY 未正确配置，无法使用聊天模型")

    elif config_type == "embedding":
        model_name = rag_conf.get("embedding_model_name")
        if model_name == "dashscope-embedding":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key or api_key == "your_dashscope_api_key_here":
                raise ConfigValidationError("DASHSCOPE_API_KEY 未正确配置，无法使用在线 Embedding")
        elif model_name == "local-embedding":
            local_path = rag_conf.get("embedding_local_path")
            if not local_path:
                raise ConfigValidationError("本地 Embedding 模型路径未配置")
            if not os.path.exists(local_path) and not _is_hf_model_id(local_path):
                raise ConfigValidationError(
                    f"本地 Embedding 模型路径不存在: {_basename(local_path)}"
                )

    elif config_type == "external_data":
        external_data_path = agent_conf.get("external_data_path")
        if not external_data_path:
            raise ConfigValidationError("external_data_path 配置项缺失")
        abs_path = get_abs_path(external_data_path)
        if not os.path.exists(abs_path):
            raise ConfigValidationError(
                f"外部数据文件不存在: {_basename(external_data_path)}"
            )
