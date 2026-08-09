"""知识库管理接口：上传文件 + 重建向量库。"""
import os
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from api.schemas import KnowledgeUploadResponse, KnowledgeRebuildResponse
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from rag.vector_store import VectorStoreService

router = APIRouter()


@router.post("/knowledge/upload", response_model=KnowledgeUploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    """上传文档到知识库目录（仅允许配置的文档类型）。"""
    data_dir = get_abs_path(chroma_conf["data_path"])
    os.makedirs(data_dir, exist_ok=True)
    allowed = tuple(chroma_conf.get("allow_knowledge_file_type", ["txt", "pdf"]))
    saved = 0
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lstrip(".").lower()
        if ext not in allowed:
            continue
        dest = os.path.join(data_dir, os.path.basename(f.filename))
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved += 1
    return KnowledgeUploadResponse(success=True, file_count=saved)


@router.post("/knowledge/rebuild", response_model=KnowledgeRebuildResponse)
async def rebuild():
    """重建向量库（基于 MD5 增量入库）。返回当前分块总数。"""
    try:
        vs = VectorStoreService()
        vs.load_document()
        try:
            chunk_count = vs.vector_store._collection.count()
        except Exception:
            chunk_count = 0
        return KnowledgeRebuildResponse(success=True, chunk_count=chunk_count)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"重建失败：{e}")
