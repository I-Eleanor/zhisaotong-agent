import copy
import json
import logging
import os
import re
from datetime import datetime

from utils.path_tool import get_abs_path

LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

_KEY_PATTERNS = [
    re.compile(r"(api[_-]?key[\"'\s:=]+)[\w\-]{8,}", re.IGNORECASE),
    re.compile(r"(token[\"'\s:=]+)[\w\-]{8,}", re.IGNORECASE),
    re.compile(r"(secret[\"'\s:=]+)[\w\-]{8,}", re.IGNORECASE),
    # 无前缀的裸密钥：sk- 开头且后接至少 8 位字符（\b 避免 task-management 之类误伤）
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{7,}"),
]

# URL 整体保护：http(s) 等链接里的 // 与 /path 段不能被 Unix 路径规则误伤；
# URL 自身先暂存，其中的密钥仍经 _redact_keys 脱敏后原样恢复
_URL_PATTERN = re.compile(
    r"[A-Za-z][A-Za-z0-9+.\-]*://[A-Za-z0-9\-._~/?#\[\]@!$&()*+,;=%]+"
)
_URL_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")

_PATH_PATTERNS = [
    # Windows 盘符绝对路径（兼容 JSON / repr 转义后的双反斜杠形态）；
    # 冒号后必须紧跟分隔符，避免误伤 "12:30" 之类时间文本
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][A-Za-z0-9._\-\\/]+"),
    # Unix 多段绝对路径；前置词边界排除相对路径（agent/diagnostic/nodes.py、
    # ../config/rag.yml 等），至少两段避免误伤孤立的短片段
    re.compile(r"(?<![A-Za-z0-9./])/[A-Za-z0-9._\-]+(?:/[A-Za-z0-9._\-]+)+"),
    # 冒号+分隔符+多段：裸密钥贪婪吞并前缀时（如 "sk-xxx_D:\a\b" 脱敏后
    # 残留 "***REDACTED***:\a\b"），盘符字母已丢失，兜底捕获残余路径形态
    re.compile(r":[\\/][A-Za-z0-9._\-]+(?:[\\/][A-Za-z0-9._\-]+)+"),
]

PATH_REDACTED = "<PATH_REDACTED>"

# 受信任应用路由白名单：结构化日志 path 字段值为固定应用路由（/api、/health 下）
# 时允许保留，用于错误定位。段内限字母数字与 - _ .（禁 .. 防目录穿越绕过）。
_ROUTE_SEGMENT = r"[A-Za-z0-9_\-]+(?:\.[A-Za-z0-9_\-]+)*"
_TRUSTED_ROUTE_PATTERN = re.compile(rf"/(?:api|health)(?:/{_ROUTE_SEGMENT})*")

ROUTE_PLACEHOLDER = "__TRUSTED_ROUTE__"


def _protect_trusted_route(msg: object) -> tuple[object, str | None]:
    """复制日志字典，把受信任应用路由的 path 字段换成占位符。

    仅顶层 path 字段且值完全匹配白名单模式时保护（返回副本与原路由）；
    其余字段（含嵌套结构、恶意构造的 path 值）一律照常走 _redact()。
    """
    if isinstance(msg, dict):
        path_value = msg.get("path")
        if isinstance(path_value, str) and _TRUSTED_ROUTE_PATTERN.fullmatch(path_value):
            patched = dict(msg)
            patched["path"] = ROUTE_PLACEHOLDER
            return patched, path_value
    return msg, None


def _redact_keys(text: str) -> str:
    for pat in _KEY_PATTERNS:
        text = pat.sub(r"\1***REDACTED***" if pat.groups else "***REDACTED***", text)
    return text


def _redact(text: str) -> str:
    stash: list[str] = []

    def _stash_url(match: re.Match[str]) -> str:
        stash.append(_redact_keys(match.group(0)))
        return f"\x00{len(stash) - 1}\x00"

    protected = _URL_PATTERN.sub(_stash_url, text)
    protected = _redact_keys(protected)
    for pat in _PATH_PATTERNS:
        protected = pat.sub(PATH_REDACTED, protected)
    return _URL_PLACEHOLDER.sub(lambda m: stash[int(m.group(1))], protected)


# 日志中用户输入的最大保留长度：超长截断，避免完整记录可能含隐私的内容
LOG_TEXT_MAX_LENGTH = 100


def log_safe_text(text: str | None, max_length: int = LOG_TEXT_MAX_LENGTH) -> str:
    """日志安全文本：压缩换行 → 脱敏（密钥 + 绝对路径）→ 截断超长输入。

    用户 query 可能包含地址、电话等隐私，日志只保留前缀用于定位问题。
    敏感文本先脱敏再截断（调用侧第一道防线；文件 / 控制台
    formatter 的 _redact 是第二道），避免截断把密钥或路径切成无法匹配的残段。
    """
    value = _redact(" ".join((text or "").split()))
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}…(共{len(value)}字)"


def safe_exception_fields(exc: Exception) -> dict[str, str]:
    """统一异常摘要字段：全项目异常日志的标准形态。

    固定返回 error_type + error_msg（脱敏、单行化、截断后的摘要），
    不含 traceback / 异常链 / 本地路径 / 完整用户输入。
    用法：logger.error({"event": "...", "stage": "...", **safe_exception_fields(e)})
    """
    return {
        "error_type": type(exc).__name__,
        "error_msg": log_safe_text(str(exc)),
    }


# 递归日志清洗的集合限额：元素数与嵌套深度（防止超大参数污染日志）
LOG_VALUE_MAX_ITEMS = 8
LOG_VALUE_MAX_DEPTH = 3


def log_safe_value(value: object, depth: int = 0) -> object:
    """递归日志清洗：只清洗日志副本，不修改原参数。

    - str → log_safe_text()（脱敏 + 截断）
    - dict / list / tuple → 递归清洗每个元素，超过 MAX_ITEMS 截断；
    - 嵌套超过 MAX_DEPTH 返回占位符（深度上限天然终止递归，循环引用不会爆栈）；
    - 基本标量原样保留；其他类型转字符串后按文本清洗（repr 可能含敏感信息）。
    """
    if isinstance(value, str):
        return log_safe_text(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if depth >= LOG_VALUE_MAX_DEPTH:
        return "<max_depth>"
    if isinstance(value, dict):
        items = list(value.items())[:LOG_VALUE_MAX_ITEMS]
        cleaned_dict = {
            log_safe_text(str(k)): log_safe_value(v, depth + 1)
            for k, v in items
        }
        if len(value) > LOG_VALUE_MAX_ITEMS:
            cleaned_dict["<truncated>"] = f"{len(value) - LOG_VALUE_MAX_ITEMS} more"
        return cleaned_dict
    if isinstance(value, list | tuple):
        cleaned_list = [log_safe_value(v, depth + 1) for v in list(value)[:LOG_VALUE_MAX_ITEMS]]
        if len(value) > LOG_VALUE_MAX_ITEMS:
            cleaned_list.append(f"<truncated {len(value) - LOG_VALUE_MAX_ITEMS} more>")
        return cleaned_list
    return log_safe_text(repr(value))


class JsonFormatter(logging.Formatter):
    def format(self, record):
        obj = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "file": f"{record.filename}:{record.lineno}",
        }
        source, route = _protect_trusted_route(record.msg)
        if isinstance(source, dict):
            obj.update(source)
        else:
            obj["message"] = record.getMessage()

        try:
            from utils.request_context import get_request_id
            rid = get_request_id()
            if rid:
                obj["request_id"] = rid
        except Exception:
            pass

        text = json.dumps(obj, ensure_ascii=False)
        text = _redact(text)
        if route is not None:
            text = text.replace(f'"{ROUTE_PLACEHOLDER}"', json.dumps(route, ensure_ascii=False))
        return text


class ConsoleFormatter(logging.Formatter):
    """控制台格式化器：输出前执行 _redact()，与文件日志（JsonFormatter）脱敏口径一致。"""

    def format(self, record: logging.LogRecord) -> str:
        source, route = _protect_trusted_route(record.msg)
        if route is None:
            return _redact(super().format(record))
        # 浅拷贝记录再替换 msg：避免污染共享 LogRecord（super().format 会缓存 record.message）
        clone = copy.copy(record)
        clone.msg = source
        text = super().format(clone)
        return _redact(text).replace(f"'{ROUTE_PLACEHOLDER}'", f"'{route}'")


CONSOLE_FORMAT = ConsoleFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(CONSOLE_FORMAT)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    return logger


logger = get_logger()
