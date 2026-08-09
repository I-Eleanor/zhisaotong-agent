from abc import ABC, abstractmethod
from typing import Optional
import os
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings, SentenceTransformerEmbeddings
from utils.config_handler import rag_conf
from utils.config_validator import validate_before_use
from langchain_openai import ChatOpenAI


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self):
        validate_before_use("chat_model")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置，请检查 .env 文件")
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=rag_conf["deepseek_base_url"],
            api_key=api_key,
            temperature=1,
        )


class EmbeddingsFactory(BaseModelFactory):
    """Embedding 模型工厂"""

    def generator(self):
        validate_before_use("embedding")
        if rag_conf["embedding_model_name"] == "local-embedding":
            local_path = rag_conf["embedding_local_path"]
            return SentenceTransformerEmbeddings(
                model_name=local_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,
                }
            )
        elif rag_conf["embedding_model_name"] == "dashscope-embedding":
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY 环境变量未设置，请检查 .env 文件")
            return DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=api_key
            )
        else:
            raise ValueError(f"不支持的 embedding 模型: {rag_conf['embedding_model_name']}")


_chat_model: Optional[BaseChatModel] = None
_embed_model: Optional[Embeddings] = None


def get_chat_model() -> BaseChatModel:
    """获取全局对话模型（懒加载，首次调用时才真正创建）。"""
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatModelFactory().generator()
    return _chat_model


def get_embed_model() -> Embeddings:
    """获取全局 Embedding 模型（懒加载，首次调用时才真正创建）。"""
    global _embed_model
    if _embed_model is None:
        _embed_model = EmbeddingsFactory().generator()
    return _embed_model


def reset_models() -> None:
    """重置模型单例，主要供测试隔离使用。"""
    global _chat_model, _embed_model
    _chat_model = None
    _embed_model = None


def __getattr__(name: str):
    """兼容旧写法 `from model.factory import chat_model`（PEP 562 模块级懒属性）。

    注意：仍推荐直接使用 get_chat_model() / get_embed_model()，
    以便在导入阶段完全不触发模型创建。
    """
    if name == "chat_model":
        return get_chat_model()
    if name == "embed_model":
        return get_embed_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
