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
        api_key = os.getenv("KIMI_API_KEY")
        if not api_key:
            raise ValueError("KIMI_API_KEY 环境变量未设置，请检查 .env 文件")
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            base_url=rag_conf["kimi_base_url"],
            api_key=api_key,
            temperature=0.7,
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


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
