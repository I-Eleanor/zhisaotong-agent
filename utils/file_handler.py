import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.logger_handler import log_safe_text, logger, safe_exception_fields


def get_file_md5_hex(filepath: str):  # 获取文件的md5的十六进制字符串

    if not os.path.exists(filepath):
        logger.error({"event": "md5_file_not_found", "file": log_safe_text(filepath)})
        return

    if not os.path.isfile(filepath):
        logger.error({"event": "md5_not_a_file", "file": log_safe_text(filepath)})
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096  # 4KB分片，避免文件过大爆内存
    try:
        with open(filepath, "rb") as f:  # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)

            """
            chunk = f.read(chunk_size)
            while chunk:

                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error({
            "event": "md5_calc_error",
            "file": log_safe_text(filepath),
            **safe_exception_fields(e),
        })
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):  # 返回文件夹内的文件列表（允许的文件后缀）
    files = []

    if not os.path.isdir(path):
        logger.error({"event": "listdir_not_a_dir", "dir": log_safe_text(path)})
        return []

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
