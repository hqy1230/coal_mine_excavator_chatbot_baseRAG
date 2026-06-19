from abc import ABC
from abc import abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from utils.config_handler import rag_conf

# 懒加载：避免模块导入时就实例化（实例化会发起网络连接，可能导致模块导入卡住）
_chat_model = None
_embed_model = None


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self):
        pass


def _get_chat_tongyi():
    from langchain_community.chat_models.tongyi import ChatTongyi
    return ChatTongyi


def _get_dashscope_embeddings():
    from langchain_community.embeddings import DashScopeEmbeddings
    return DashScopeEmbeddings


class ChatModelFactory(BaseModelFactory):
    def generator(self):
        ChatTongyi = _get_chat_tongyi()
        return ChatTongyi(
            model=rag_conf["chat_model_name"],
            streaming=True,
            temperature=0.1,
            max_tokens=1000,  # 限制回答长度约500字
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self):
        DashScopeEmbeddings = _get_dashscope_embeddings()
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


def get_chat_model():
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatModelFactory().generator()
    return _chat_model


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = EmbeddingsFactory().generator()
    return _embed_model


# 向后兼容：旧的 embed_model 名字（懒代理）
class _LazyProxy:
    def __getattr__(self, name):
        return getattr(get_embed_model(), name)


embed_model = _LazyProxy()
chat_model = _LazyProxy()  # 仅占位，不应被调用
