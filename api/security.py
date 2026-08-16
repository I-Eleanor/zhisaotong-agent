"""API 安全中间件：限流、可选 Token 鉴权、统一错误响应。"""
import os

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from utils.logger_handler import logger

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

API_TOKEN = os.getenv("API_TOKEN", "")
ADMIN_PATHS = {"/api/knowledge/rebuild", "/api/knowledge/upload"}


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """可选 API Token 鉴权：设置了 API_TOKEN 环境变量时，管理接口需要 Bearer Token。"""

    async def dispatch(self, request: Request, call_next):
        if not API_TOKEN:
            return await call_next(request)

        if request.url.path not in ADMIN_PATHS:
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            logger.warning({"event": "auth_missing", "path": request.url.path})
            return JSONResponse(status_code=401, content={"detail": "缺少 Authorization 头"})

        token = auth[7:]
        if token != API_TOKEN:
            logger.warning({"event": "auth_failed", "path": request.url.path})
            return JSONResponse(status_code=403, content={"detail": "Token 无效"})

        return await call_next(request)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """请求体大小限制。"""

    MAX_BODY_SIZE = 50 * 1024 * 1024

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            return JSONResponse(status_code=413, content={"detail": "请求体过大"})
        return await call_next(request)
