import json
import logging
import os
import re
from datetime import datetime

from utils.path_tool import get_abs_path

LOG_ROOT = get_abs_path("logs")
os.makedirs(LOG_ROOT, exist_ok=True)

_SENSITIVE_PATTERNS = [
    re.compile(r"(api[_-]?key[\"'\s:=]+)[\w\-]{8,}", re.IGNORECASE),
    re.compile(r"(token[\"'\s:=]+)[\w\-]{8,}", re.IGNORECASE),
    re.compile(r"(secret[\"'\s:=]+)[\w\-]{8,}", re.IGNORECASE),
]


def _redact(text: str) -> str:
    for pat in _SENSITIVE_PATTERNS:
        text = pat.sub(r"\1***REDACTED***", text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record):
        obj = {
            "ts": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "file": f"{record.filename}:{record.lineno}",
        }
        msg = record.getMessage()
        if isinstance(record.msg, dict):
            obj.update(record.msg)
        else:
            obj["message"] = msg

        try:
            from utils.request_context import get_request_id
            rid = get_request_id()
            if rid:
                obj["request_id"] = rid
        except Exception:
            pass

        text = json.dumps(obj, ensure_ascii=False)
        return _redact(text)


CONSOLE_FORMAT = logging.Formatter(
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
