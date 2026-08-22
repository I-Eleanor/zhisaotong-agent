"""FastAPI 应用入口。

启动：uvicorn api.main:app --host 0.0.0.0 --port 8000
API 文档：http://localhost:8000/docs
"""
import asyncio
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.container import AppContainer, ContainerState, get_mounted_container
from api.routes import conversation, diagnostic, knowledge
from api.security import RequestSizeLimitMiddleware, TokenAuthMiddleware, limiter
from api.streaming import shutdown_sse_executor
from utils import error_codes
from utils.config_handler import chroma_conf, rag_conf
from utils.config_validator import ConfigValidationError, validate_startup
from utils.exceptions import AgentProjectError, safe_error_payload
from utils.logger_handler import log_safe_text, logger, safe_exception_fields
from utils.request_context import get_request_id, new_request_id, set_request_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动前配置校验（失败即终止），通过后新建容器挂到 app.state。

    启动顺序（P1-13.1 安全边界）：
    1. validate_startup() 在容器创建前执行：纯配置检查（环境变量 / 路径 /
       模型配置 / CORS 格式），不加载模型、Embedding、Chroma；失败时记录
       结构化安全日志（统一 safe_exception_fields 异常摘要形态）并
       抛出异常阻止启动，不构造、不挂载新容器。
    2. AppContainer 构造本身零成本（资源全部懒加载），启动阶段不加载
       模型 / Embedding / Chroma。
    关闭时使用局部引用而非 app.state.container：若运行期间后者被测试 /
    重载逻辑替换，仍只关闭本次 lifespan 创建的实例。
    """
    logger.info({"event": "app_startup"})
    try:
        validate_startup()
    except ConfigValidationError as exc:
        logger.error({
            "event": "startup_config_validation_failed",
            **safe_exception_fields(exc),
        })
        raise
    container = AppContainer()
    app.state.container = container
    try:
        yield
    finally:
        container.close()
        shutdown_sse_executor(wait=True)
        logger.info({"event": "app_shutdown"})


app = FastAPI(title="智扫通 Agent API", version="2.0.0", lifespan=lifespan)

app.state.limiter = limiter
# slowapi 处理器签名 (Request, RateLimitExceeded) 与 Starlette 期望的 (Request, Exception) 不匹配，
# 属库类型标注问题，运行时无影响
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


# 统一错误响应：固定 500 + 安全载荷 + 回写 X-Request-ID
# （未知异常经 ServerErrorMiddleware 返回时，用户层中间件已被跳过，头部需自行补上）
def _error_response(exc: Exception, rid: str) -> JSONResponse:
    response = JSONResponse(status_code=500, content=safe_error_payload(exc, request_id=rid))
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(AgentProjectError)
async def agent_project_error_handler(request: Request, exc: AgentProjectError) -> JSONResponse:
    """项目异常统一转换：使用异常自身的 error_code 与安全提示，细节只进日志。

    不记录完整 traceback（可能携带用户输入、本地绝对路径与异常链原文）；
    error_msg / original_error_msg 均经 log_safe_text 脱敏、单行化并截断。
    """
    rid = get_request_id()
    log_data: dict = {
        "event": "unhandled_project_error",
        "request_id": rid,
        "path": request.url.path,
        "error_code": exc.error_code,
        "stage": exc.stage,
        "error_type": type(exc).__name__,
        "error_msg": log_safe_text(str(exc)),
    }
    if exc.original is not None:
        log_data["original_error_type"] = type(exc.original).__name__
        log_data["original_error_msg"] = log_safe_text(str(exc.original))
    logger.warning(log_data)
    return _error_response(exc, rid)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """未预期异常：固定 500 / INTERNAL_ERROR / 通用安全提示，原始异常只进日志。

    同样不记录完整 traceback，只保留清洗后的 error_msg 摘要。
    """
    rid = get_request_id()
    logger.error({
        "event": "unhandled_exception",
        "request_id": rid,
        "path": request.url.path,
        "stage": "api",
        "error_type": type(exc).__name__,
        "error_msg": log_safe_text(str(exc)),
    })
    return _error_response(exc, rid)


def _parse_cors_origins() -> list[str]:
    """解析 CORS_ORIGINS 为来源列表；通配符 * 时记录安全警告。

    通配符允许任意来源跨域访问（含带凭据场景的浏览器默认策略），
    仅适合本地开发；生产环境应在环境变量中配置显式来源列表
    （逗号分隔的完整 origin），格式校验见 validate_cors_origins()。
    开发兼容性不变：默认值仍为 *。
    """
    raw = os.getenv("CORS_ORIGINS", "*")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        logger.warning({
            "event": "cors_wildcard_enabled",
            "stage": "security",
            "hint": "CORS_ORIGINS=* 允许任意来源跨域访问，生产环境应配置显式来源列表",
        })
    return origins


_cors_origins_list = _parse_cors_origins()
app.add_middleware(CORSMiddleware, allow_origins=_cors_origins_list, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TokenAuthMiddleware)
app.add_middleware(RequestSizeLimitMiddleware)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", "") or new_request_id()
    set_request_id(rid)
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    logger.info({
        "event": "request_completed",
        "request_id": rid,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": duration_ms,
    })
    return response


app.include_router(conversation.router, prefix="/api")
app.include_router(diagnostic.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")


@app.get("/api/health")
async def health():
    """健康检查：返回服务状态。"""
    return {
        "status": "ok",
        "model": rag_conf.get("chat_model_name", "unknown"),
        "reranker_enabled": chroma_conf.get("reranker_enabled", False),
    }


@app.get("/api/health/live")
async def health_live():
    """存活探针：进程是否存活。"""
    return {"status": "alive"}


# 就绪检查单项的超时（秒）：远端依赖慢时快速失败，不长时间阻塞探针
READY_CHECK_TIMEOUT_SECONDS = 3.0


@app.get("/api/health/ready")
async def health_ready(container: AppContainer = Depends(get_mounted_container)):  # noqa: B008
    """就绪探针：容器 OPEN 且必要依赖（向量库、模型）可用时返回 200，否则 503。

    非就绪统一 503（P1-16）：
    - 容器未挂载：Depends(get_mounted_container) 抛 503 并记 container_not_ready 日志；
    - 容器 CLOSING / CLOSED：直接 503，不执行资源检查（不触发懒加载重建），
      记 container_not_ready 日志（event / state / request_id / error_code 安全字段）；
    - 必要资源检查失败 / 超时：503。

    检查语义（浅 / 深）：
    - vector_store：深检查，实际读取集合计数（本地 Chroma，读取即验证可用）；
    - chat_model：浅检查，仅验证模型对象可创建 / 已缓存，不做远端调用探测
      （避免每次探针产生真实 LLM 请求费用与延迟）。
    每项检查经线程池执行并带短超时，同步调用不阻塞事件循环。
    容器经 Depends 注入（lifespan 挂载或测试夹具注入），不回退全局容器。
    """
    state = container.state
    if state is not ContainerState.OPEN:
        logger.warning({
            "event": "container_not_ready",
            "state": state.value,
            "request_id": get_request_id(),
            "error_code": error_codes.CONTAINER_NOT_READY,
        })
        return JSONResponse(status_code=503, content={"status": "not_ready", "checks": {}})

    checks: dict[str, dict] = {}

    async def run_check(name: str, fn) -> None:
        try:
            result = await asyncio.wait_for(
                run_in_threadpool(fn), timeout=READY_CHECK_TIMEOUT_SECONDS
            )
            checks[name] = {"ok": True, **(result or {})}
        except TimeoutError:
            checks[name] = {"ok": False, "error": f"检查超时（{READY_CHECK_TIMEOUT_SECONDS}s）"}
        except Exception as e:
            logger.warning({
                "event": "health_check_failed",
                "check": name,
                "error_type": type(e).__name__,
            })
            checks[name] = {"ok": False, "error": "依赖不可用"}

    def check_vector_store() -> dict:
        return {"chunk_count": container.vector_store.count()}

    def check_chat_model() -> dict:
        _ = container.chat_model  # 浅检查：能拿到模型对象即可
        return {"mode": "shallow"}

    await run_check("vector_store", check_vector_store)
    await run_check("chat_model", check_chat_model)

    all_ok = all(c.get("ok", False) for c in checks.values())
    body = {"status": "ready" if all_ok else "not_ready", "checks": checks}
    return JSONResponse(status_code=200 if all_ok else 503, content=body)
