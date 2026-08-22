"""知识库管理接口：上传文件 + 重建向量库 + 带来源查询。"""
import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from api.container import AppContainer, get_app_container
from api.schemas import KnowledgeRebuildResponse, KnowledgeUploadResponse
from api.security import limiter
from utils.config_handler import chroma_conf
from utils.exceptions import AgentProjectError
from utils.logger_handler import log_safe_text, log_safe_value, logger
from utils.path_tool import get_abs_path

router = APIRouter()

ALLOWED_EXTENSIONS = tuple(chroma_conf.get("allow_knowledge_file_type", ["txt", "pdf"]))
ALLOWED_MIME_TYPES = {"text/plain", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_FILE_COUNT = 10
MAX_FILENAME_LENGTH = 255
MAX_TEXT_LENGTH = 5 * 1024 * 1024


def _validate_upload_file(f: UploadFile) -> str | None:
    if not f.filename:
        return "文件名不能为空"
    if len(f.filename) > MAX_FILENAME_LENGTH:
        return f"文件名过长（最大 {MAX_FILENAME_LENGTH} 字符）"
    ext = os.path.splitext(f.filename)[1].lstrip(".").lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"不支持的文件类型: .{ext}"
    if f.content_type and f.content_type not in ALLOWED_MIME_TYPES and not f.content_type.startswith("text/"):
        return f"不支持的 MIME 类型: {f.content_type}"
    basename = os.path.basename(f.filename)
    if basename != f.filename:
        return "文件名包含非法路径字符"
    return None


@router.post("/knowledge/upload", response_model=KnowledgeUploadResponse)
@limiter.limit("10/minute")
async def upload(request: Request, files: list[UploadFile] = File(...)):  # noqa: B008
    """上传文档到知识库目录（仅允许配置的文档类型）。"""
    if len(files) > MAX_FILE_COUNT:
        raise HTTPException(status_code=400, detail=f"最多上传 {MAX_FILE_COUNT} 个文件")

    data_dir = get_abs_path(chroma_conf["data_path"])
    os.makedirs(data_dir, exist_ok=True)

    saved = 0
    errors = []
    for f in files:
        err = _validate_upload_file(f)
        if err:
            errors.append(f"{f.filename or '未知'}: {err}")
            continue

        content = await f.read()
        if len(content) == 0:
            errors.append(f"{f.filename}: 空文件")
            continue
        if len(content) > MAX_FILE_SIZE:
            errors.append(f"{f.filename}: 文件过大（最大 {MAX_FILE_SIZE // 1024 // 1024}MB）")
            continue
        if f.filename and f.filename.endswith(".txt") and len(content) > MAX_TEXT_LENGTH:
            errors.append(f"{f.filename}: 文本内容过长")
            continue

        filename = f.filename or ""
        ext = os.path.splitext(filename)[1].lstrip(".").lower() if filename else ""
        safe_name = f"{uuid.uuid4().hex[:8]}.{ext}"
        dest = os.path.join(data_dir, safe_name)

        with open(dest, "wb") as out:
            out.write(content)
        saved += 1
        logger.info({
            "event": "file_uploaded",
            "original_name": log_safe_text(f.filename),
            "stored_name": safe_name,
            "size": len(content),
        })

    if errors:
        logger.warning({"event": "upload_errors", "errors": log_safe_value(errors)})

    return KnowledgeUploadResponse(success=True, file_count=saved)


@router.post("/knowledge/rebuild", response_model=KnowledgeRebuildResponse)
@limiter.limit("5/minute")
async def rebuild(request: Request, container: AppContainer = Depends(get_app_container)):  # noqa: B008
    """重建向量库（基于 MD5 增量入库）。返回当前分块总数。

    业务异常交给全局 handler 统一转换；未知异常（Chroma / 文件系统等）
    包装成 AgentProjectError 后抛出，响应不携带任何原始异常文本。
    """
    try:
        vs = container.vector_store
        vs.load_document()
        try:
            chunk_count = vs.count()
        except Exception:
            chunk_count = 0
        return KnowledgeRebuildResponse(success=True, chunk_count=chunk_count)
    except AgentProjectError:
        raise
    except Exception as e:
        raise AgentProjectError("知识库重建失败", stage="knowledge_rebuild", original=e) from e


class KnowledgeQueryRequest(BaseModel):
    query: str


@router.post("/knowledge/query")
async def query_with_sources(
    request: KnowledgeQueryRequest,
    container: AppContainer = Depends(get_app_container),  # noqa: B008
):
    """带来源的 RAG 查询，返回答案和结构化来源信息。

    业务异常交给全局 handler 统一转换；未知异常包装成 AgentProjectError 后抛出。
    """
    try:
        rag = container.rag_service
        result = rag.rag_with_sources(request.query)
        return result
    except AgentProjectError:
        raise
    except Exception as e:
        raise AgentProjectError("知识库查询失败", stage="knowledge_query", original=e) from e
