"""人工转接工单接口。

用户侧：
- POST  /api/handoff
        创建工单（pending），生成随机 access_key；
        响应返回完整工单 + 明文 access_key（仅此一次，库中只存哈希）
- GET   /api/handoff/{ticket_id}?access_key=xxx
        用户凭 access_key 查询自己的工单（常量时间哈希比较）；
        缺失 / 错误 / 工单不存在一律 404，不泄露存在性

conversation_id 仅用于会话关联展示，不再作为用户访问凭证。

管理端（会话 Cookie 鉴权，需设置 ADMIN_TOKEN 环境变量；未配置时接口整体禁用 403）：
- POST  /api/admin/login                          登录，提交 ADMIN_TOKEN，成功下发 HttpOnly 会话 Cookie
- GET   /api/admin/handoffs?status=&limit=&offset=   分页列表
- GET   /api/admin/handoffs/{ticket_id}              详情
- POST  /api/admin/handoffs/{ticket_id}/reply        保存 human_reply 并更新状态

管理端鉴权一律校验服务端会话（Cookie admin_session），旧版请求头鉴权方式已弃用。
所有接口均不回显 access_key / access_key_hash。
"""
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from api.admin_session import (
    SESSION_COOKIE_NAME,
    admin_enabled,
    admin_token_matches,
    session_store,
    session_ttl_seconds,
)
from api.handoff_store import VALID_STATUSES, HandoffStore, HandoffStoreError
from api.schemas import AdminLoginRequest, HandoffCreateRequest, HandoffReplyRequest
from utils import error_codes
from utils.logger_handler import log_safe_text, logger
from utils.request_context import get_request_id

router = APIRouter(tags=["handoff"])

# 管理端列表分页上限
HANDOFF_LIST_LIMIT_MAX = 100


def _get_store() -> HandoffStore | JSONResponse:
    """构造存储实例；环境变量未配置时返回统一的 503 响应。"""
    try:
        return HandoffStore()
    except HandoffStoreError as e:
        logger.warning({
            "event": "handoff_store_unavailable",
            "request_id": get_request_id(),
            "error_msg": log_safe_text(str(e)),
        })
        return JSONResponse(
            status_code=503,
            content={
                "error_code": error_codes.HANDOFF_DB_NOT_CONFIGURED,
                "safe_message": "工单服务未配置，请联系管理员。",
                "request_id": get_request_id(),
            },
        )


def _admin_guard(request: Request) -> JSONResponse | None:
    """管理接口鉴权：ADMIN_TOKEN 未配置 → 403 禁用；无/无效/过期会话 Cookie → 401。"""
    if not admin_enabled():
        logger.warning({
            "event": "handoff_admin_disabled",
            "request_id": get_request_id(),
        })
        return JSONResponse(
            status_code=403,
            content={
                "error_code": error_codes.HANDOFF_ADMIN_DISABLED,
                "safe_message": "管理接口未启用。",
                "request_id": get_request_id(),
            },
        )
    sid = request.cookies.get(SESSION_COOKIE_NAME, "")
    if not sid or not session_store.get(sid):
        logger.warning({"event": "handoff_admin_auth_failed", "request_id": get_request_id()})
        return JSONResponse(
            status_code=401,
            content={
                "error_code": error_codes.HANDOFF_ADMIN_REQUIRED,
                "safe_message": "需要有效的管理会话。",
                "request_id": get_request_id(),
            },
        )
    return None


def _not_found(rid: str) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error_code": error_codes.HANDOFF_TICKET_NOT_FOUND,
            "safe_message": "工单不存在。",
            "request_id": rid,
        },
    )


def _store_error(e: HandoffStoreError) -> JSONResponse:
    logger.error({
        "event": "handoff_store_error",
        "request_id": get_request_id(),
        "error_msg": log_safe_text(str(e)),
    })
    return JSONResponse(
        status_code=500,
        content={
            "error_code": error_codes.INTERNAL_ERROR,
            "safe_message": "工单服务暂时异常，请稍后重试。",
            "request_id": get_request_id(),
        },
    )


def _validate_status(status: str | None) -> str | None:
    """查询参数 status 合法性校验；非法返回 None（由调用方转 422）。"""
    if status is None or status in VALID_STATUSES:
        return status
    return None


@router.post("/admin/login")
def admin_login(body: AdminLoginRequest):
    """管理端登录。ADMIN_TOKEN 校验通过后建立服务端会话并下发 HttpOnly Cookie。

    ADMIN_TOKEN 未配置 → 403 禁用；令牌错误 → 401。响应不回显令牌。
    """
    rid = get_request_id()
    if not admin_enabled():
        logger.warning({
            "event": "handoff_admin_disabled",
            "request_id": rid,
        })
        return JSONResponse(
            status_code=403,
            content={
                "error_code": error_codes.HANDOFF_ADMIN_DISABLED,
                "safe_message": "管理接口未启用。",
                "request_id": rid,
            },
        )
    if not admin_token_matches(body.admin_token):
        logger.warning({"event": "handoff_admin_login_failed", "request_id": rid})
        return JSONResponse(
            status_code=401,
            content={
                "error_code": error_codes.HANDOFF_ADMIN_REQUIRED,
                "safe_message": "管理员令牌无效。",
                "request_id": rid,
            },
        )
    sid = session_store.create(session_ttl_seconds())
    ttl = session_ttl_seconds()
    logger.info({"event": "handoff_admin_login_success", "request_id": rid})
    response = JSONResponse(
        status_code=200,
        content={"success": True, "message": "登录成功", "request_id": rid},
    )
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=sid,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=ttl,
        path="/",
    )
    return response


@router.post("/handoff")
def create_handoff(body: HandoffCreateRequest):
    """用户提交人工转接工单。返回完整工单（status=pending）+ 明文 access_key。"""
    rid = get_request_id()
    store = _get_store()
    if isinstance(store, JSONResponse):
        return store
    try:
        ticket, access_key = store.create(
            user_question=body.user_question.strip(),
            conversation_id=(body.conversation_id or "").strip() or None,
            recent_conversations=body.recent_conversations,
            conversation_summary=body.conversation_summary,
            retrieval_sources=body.retrieval_sources,
            diagnostic_result=body.diagnostic_result,
            handoff_reason=body.handoff_reason,
        )
    except HandoffStoreError as e:
        return _store_error(e)
    logger.info({
        "event": "handoff_ticket_created",
        "ticket_id": ticket["ticket_id"],
        "conversation_id": ticket["conversation_id"],
        "request_id": rid,
    })
    # 明文 access_key 仅在创建响应中出现一次；后续任何接口不再返回
    return JSONResponse(status_code=201, content={**ticket, "access_key": access_key})


@router.get("/handoff/{ticket_id}")
def get_handoff(ticket_id: str, access_key: str | None = Query(default=None)):
    """用户凭 access_key 查询工单；缺失 / 错误 / 不存在统一 404。"""
    store = _get_store()
    if isinstance(store, JSONResponse):
        return store
    try:
        ticket = store.get_by_access_key(ticket_id, access_key or "")
    except HandoffStoreError as e:
        return _store_error(e)
    if ticket is None:
        # 三种失败同响应：缺失凭证 / 错误凭证 / 工单不存在，不泄露存在性
        logger.warning({
            "event": "handoff_user_get_rejected",
            "ticket_id": ticket_id,
            "has_access_key": bool(access_key),
            "request_id": rid,
        })
        return _not_found(rid)
    return ticket


@router.get("/admin/handoffs")
def list_handoffs(
    request: Request,
    status: str | None = None,
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
):
    """管理端分页查询工单，可按状态过滤，created_at 倒序。"""
    guard = _admin_guard(request)
    if guard is not None:
        return guard
    if limit > HANDOFF_LIST_LIMIT_MAX:
        limit = HANDOFF_LIST_LIMIT_MAX
    rid = get_request_id()
    store = _get_store()
    if isinstance(store, JSONResponse):
        return store
    if status is not None and _validate_status(status) is None:
        return JSONResponse(
            status_code=422,
            content={"detail": f"status 必须是 {'/'.join(sorted(VALID_STATUSES))} 之一"},
        )
    try:
        total, items = store.list_tickets(status=status, limit=limit, offset=offset)
    except HandoffStoreError as e:
        return _store_error(e)
    return {"total": total, "limit": limit, "offset": offset, "items": items}


@router.get("/admin/handoffs/{ticket_id}")
def admin_get_handoff(request: Request, ticket_id: str):
    """管理端查询单个工单详情。"""
    guard = _admin_guard(request)
    if guard is not None:
        return guard
    rid = get_request_id()
    store = _get_store()
    if isinstance(store, JSONResponse):
        return store
    try:
        ticket = store.get(ticket_id)
    except HandoffStoreError as e:
        return _store_error(e)
    if ticket is None:
        return _not_found(rid)
    return ticket


@router.post("/admin/handoffs/{ticket_id}/reply")
def reply_handoff(request: Request, ticket_id: str, body: HandoffReplyRequest):
    """管理端回复工单：保存 human_reply 并更新状态（缺省置为 resolved）。"""
    guard = _admin_guard(request)
    if guard is not None:
        return guard
    rid = get_request_id()
    store = _get_store()
    if isinstance(store, JSONResponse):
        return store
    try:
        ticket = store.reply(
            ticket_id,
            human_reply=body.human_reply.strip(),
            status=body.status or "resolved",
        )
    except HandoffStoreError as e:
        return _store_error(e)
    if ticket is None:
        return _not_found(rid)
    logger.info({
        "event": "handoff_ticket_replied",
        "ticket_id": ticket_id,
        "status": ticket.get("status"),
        "request_id": rid,
    })
    return ticket
