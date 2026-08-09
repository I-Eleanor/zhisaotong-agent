import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import get_embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self, embedding_function=None, persist_directory: str = None, collection_name: str = None):
        self.vector_store = Chroma(
            collection_name=collection_name or chroma_conf["collection_name"],
            embedding_function=embedding_function or get_embed_model(),
            persist_directory=persist_directory or chroma_conf["persist_directory"],
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    @staticmethod
    def _build_source_filter(source_files: list[str] | tuple[str, ...] | str | None) -> dict | None:
        """构造 Chroma 元数据过滤条件，用于把检索范围限定在指定知识库文件内。"""
        if not source_files:
            return None

        if isinstance(source_files, str):
            source_files = [source_files]

        source_files = [s for s in source_files if s]
        if not source_files:
            return None

        if len(source_files) == 1:
            return {"source_file": source_files[0]}

        return {"source_file": {"$in": list(source_files)}}

    def get_retriever(self, k: int = None, source_files: list[str] | str = None):
        """获取检索器。

        :param k: 召回数量，默认取配置中的 k
        :param source_files: 可选，将检索范围限定在指定的知识库文件（用于诊断类定向检索）
        """
        search_kwargs = {"k": k or chroma_conf["k"]}

        source_filter = self._build_source_filter(source_files)
        if source_filter:
            search_kwargs["filter"] = source_filter

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    @staticmethod
    def _enrich_metadata(documents: list[Document], file_path: str, chunk_index_offset: int = 0) -> list[Document]:
        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_name)[1].lstrip(".")
        for i, doc in enumerate(documents):
            doc.metadata["source_file"] = file_name
            doc.metadata["file_type"] = file_type
            doc.metadata["chunk_index"] = chunk_index_offset + i
            if "page" not in doc.metadata:
                doc.metadata["page"] = -1
        return documents

    def load_document(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                self._enrich_metadata(documents, path)

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                for i, doc in enumerate(split_document):
                    doc.metadata["chunk_index"] = i

                self.vector_store.add_documents(split_document)

                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功，共{len(split_document)}个分块")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print(r.metadata)
        print("-" * 20)