"""轻量管理端会话（服务端内存会话 + HttpOnly Cookie）。

管理端登录后，服务端为每个会话分配高熵 session_id 存入内存字典，
浏览器侧仅保存一个不透明 HttpOnly Cookie（Secure / SameSite / Max-Age），
前端 JS 无法读取会话值。所有 /api/admin/* 接口改为校验该 Cookie 对应的
服务端会话，旧版请求头鉴权方式已弃用。

设计要点：
- 会话存于单进程内存：服务重启即失效（需重新登录），符合轻量定位；
- 有效期由 ADMIN_SESSION_TTL_SECONDS 控制（默认 8 小时）；
- ADMIN_TOKEN 未配置时，登录与管理接口整体禁用（403）；
- 本模块不暴露 / 存储明文 ADMIN_TOKEN，仅提供常量时间比对。
"""
import os
import secrets
import threading
import time

SESSION_COOKIE_NAME = "admin_session"

# 默认会话有效期（秒）：8 小时
DEFAULT_SESSION_TTL_SECONDS = 8 * 3600


def _admin_token() -> str:
    """读取环境变量中的管理令牌（每次读取，便于测试动态切换）。"""
    return os.getenv("ADMIN_TOKEN", "").strip()


def admin_enabled() -> bool:
    """管理端是否启用：ADMIN_TOKEN 非空才算启用。"""
    return bool(_admin_token())


def admin_token_matches(provided: str) -> bool:
    """常量时间比对登录提交的令牌与 ADMIN_TOKEN；任一为空返回 False。"""
    expected = _admin_token()
    if not expected or not provided:
        return False
    return secrets.compare_digest(provided, expected)


def session_ttl_seconds() -> int:
    """会话有效期（秒），由 ADMIN_SESSION_TTL_SECONDS 配置。"""
    raw = os.getenv("ADMIN_SESSION_TTL_SECONDS", "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_SESSION_TTL_SECONDS


class AdminSessionStore:
    """服务端会话表：session_id -> 过期时间戳。内存实现，线程安全。"""

    def __init__(self) -> None:
        self._sessions: dict[str, float] = {}
        self._lock = threading.Lock()

    def create(self, ttl_seconds: int) -> str:
        """新建会话并返回高熵 session_id。"""
        sid = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[sid] = time.time() + ttl_seconds
        return sid

    def get(self, sid: str) -> bool:
        """会话是否存在且未过期；过期会话会被惰性清除。"""
        if not sid:
            return False
        now = time.time()
        with self._lock:
            expires = self._sessions.get(sid)
            if expires is None:
                return False
            if expires <= now:
                self._sessions.pop(sid, None)
                return False
            return True

    def revoke(self, sid: str) -> None:
        """登出：删除会话。"""
        with self._lock:
            self._sessions.pop(sid, None)

    def force_expire(self, sid: str) -> None:
        """测试辅助：将会话直接置为过期。"""
        with self._lock:
            self._sessions[sid] = time.time() - 1

    def clear(self) -> None:
        """清空所有会话（测试隔离用）。"""
        with self._lock:
            self._sessions.clear()


# 全局会话存储（单进程）
session_store = AdminSessionStore()