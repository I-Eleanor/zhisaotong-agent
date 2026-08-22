import os
import threading
from abc import ABC, abstractmethod

from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings, SentenceTransformerEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from utils.config_handler import rag_conf
from utils.config_validator import validate_before_use
from utils.exceptions import ModelInvocationError
from utils.logger_handler import logger, safe_exception_fields


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Embeddings | BaseChatModel | None:
        pass


def _build_chat_model():
    """构造聊天模型实例；密钥只经环境变量读取，构造失败不泄漏密钥。

    配置校验失败（ConfigValidationError / 密钥未设置的 ValueError）原样上抛：
    消息本身只含变量名与提示，不含密钥原文。
    """
    validate_before_use("chat_model")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置，请检查 .env 文件")
    try:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=rag_conf["deepseek_base_url"],
            api_key=api_key,
            temperature=1,
            request_timeout=float(rag_conf.get("llm_timeout_seconds", 60)),
        )
    except Exception as e:
        # 构造失败（SDK/网络/认证）：客户端只见安全提示，异常细节只进脱敏日志
        logger.error({"event": "chat_model_init_error", "stage": "model", **safe_exception_fields(e)})
        raise ModelInvocationError("聊天模型初始化失败", original=e) from e


def _build_embedding_model():
    """构造 Embedding 模型实例；密钥只经环境变量读取，构造失败不泄漏密钥。"""
    validate_before_use("embedding")
    if rag_conf["embedding_model_name"] == "local-embedding":
        # 优先环境变量（本地开发用已下载模型），需验证路径存在
        # 路径不存在时（如 Docker 容器内）自动回退到 config 中的 HF 模型名
        local_path = os.getenv("LOCAL_EMBEDDING_PATH")
        if not (local_path and os.path.exists(local_path)):
            local_path = rag_conf["embedding_local_path"]
        try:
            return SentenceTransformerEmbeddings(
                model_name=local_path,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,
                }
            )
        except Exception as e:
            logger.error({"event": "embedding_init_error", "stage": "model", **safe_exception_fields(e)})
            raise ModelInvocationError("Embedding 模型初始化失败", original=e) from e
    elif rag_conf["embedding_model_name"] == "dashscope-embedding":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置，请检查 .env 文件")
        try:
            return DashScopeEmbeddings(
                model="text-embedding-v2",
                dashscope_api_key=api_key
            )
        except Exception as e:
            logger.error({"event": "embedding_init_error", "stage": "model", **safe_exception_fields(e)})
            raise ModelInvocationError("Embedding 模型初始化失败", original=e) from e
    else:
        raise ValueError(f"不支持的 embedding 模型: {rag_conf['embedding_model_name']}")


class ChatModelFactory(BaseModelFactory):
    def generator(self):
        return _build_chat_model()


class EmbeddingsFactory(BaseModelFactory):
    """Embedding 模型工厂"""

    def generator(self):
        return _build_embedding_model()


_chat_model: BaseChatModel | None = None
_embed_model: Embeddings | None = None
_model_lock = threading.Lock()


def get_chat_model() -> BaseChatModel:
    """获取全局对话模型（懒加载 + 双检锁，并发首次调用只创建一次）。"""
    global _chat_model
    if _chat_model is None:
        with _model_lock:
            if _chat_model is None:
                _chat_model = ChatModelFactory().generator()
    return _chat_model


def get_embed_model() -> Embeddings:
    """获取全局 Embedding 模型（懒加载 + 双检锁，并发首次调用只创建一次）。"""
    global _embed_model
    if _embed_model is None:
        with _model_lock:
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
