from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.logger_handler import logger
from utils.config_handler import chroma_conf, prompts_conf
from utils.path_tools import get_abs_path
import threading

# SFT和RL支持：通过懒加载和预计算优化性能
class RagSummarizeService:
    _PROMPT_TEXT: str = None
    _sft_store = None
    _rl_policy = None
    _sft_loaded = False
    _rl_loaded = False

    def __init__(self, vector_store: VectorStoreService):
        from model.factory import get_chat_model
        self.vector_store = vector_store
        self.prompt_text = self._load_prompt_text()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()  # 延迟加载模型
        self.chain = self._init_chain()
        # 后台预加载SFT和RL（不阻塞启动）
        self._preload_sft_rl()

    def _preload_sft_rl(self):
        """后台预加载SFT和RL，不影响启动速度"""
        def load():
            try:
                from optimization.sft_store import sft_store
                self._sft_store = sft_store
                # 触发预计算嵌入
                self._sft_store._ensure_embeddings()
                self._sft_loaded = True
                logger.info("[RAG] SFT store preloaded")
            except Exception as e:
                logger.warning(f"[RAG] SFT preload failed: {e}")
                
            try:
                from optimization.rl_retrieval_policy import rl_retrieval_policy
                self._rl_policy = rl_retrieval_policy
                self._rl_loaded = True
                logger.info("[RAG] RL policy preloaded")
            except Exception as e:
                logger.warning(f"[RAG] RL preload failed: {e}")
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def _load_prompt_text(self) -> str:
        if self._PROMPT_TEXT is not None:
            return self._PROMPT_TEXT

        path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
        try:
            with open(path, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
        except Exception as e:
            logger.error(f"读取提示词文件失败：{str(e)}")
            raise RuntimeError(f"读取提示词文件失败：{str(e)}")

        if not prompt_text:
            raise ValueError(f"提示词文件内容为空：{path}")

        self._PROMPT_TEXT = prompt_text
        return prompt_text

    def _init_chain(self):
        return self.prompt_template | self.model | StrOutputParser()

    def retrieve_docs(self, query: str, k: int | None = None) -> list[Document]:
        return self.vector_store.get_retriever(k=k).invoke(query)

    def rag_summarize(self, query: str, enable_sft=True, enable_rl=True) -> str:
        # RL选择最优k值（快速决策）
        k = int(chroma_conf["k"])
        action_idx = None
        
        if enable_rl and self._rl_loaded and self._rl_policy:
            try:
                k, action_idx = self._rl_policy.choose_k(query)
            except Exception as e:
                logger.warning(f"[RL] choose_k failed: {e}")
                k = int(chroma_conf["k"])
        
        # 检索文档
        context_docs = self.retrieve_docs(query, k=k)
        
        # 构建上下文
        context = ""
        for i, doc in enumerate(context_docs):
            context += f"资料{i+1}：{doc.page_content}\n"

        # SFT样例注入（预加载的嵌入）
        sft_block = ""
        if enable_sft and self._sft_loaded and self._sft_store:
            try:
                sft_block = self._sft_store.format_exemplar_block(query)
            except Exception as e:
                logger.warning(f"[SFT] format_exemplar_block failed: {e}")
                sft_block = ""
        else:
            sft_block = ""

        # 构建输入
        input_dict = {
            "input": query,
            "context": context,
            "sft_exemplars": sft_block,
        }
        
        # 生成回答
        reply = self.chain.invoke(input_dict)
        
        # RL奖励更新（后台异步执行，不阻塞输出）
        if enable_rl and self._rl_loaded and self._rl_policy and action_idx is not None:
            self._async_rl_observe(query, action_idx, context, reply)
        
        return reply

    def _async_rl_observe(self, query, action_idx, context, reply):
        """异步更新RL策略，不影响响应速度"""
        def observe():
            try:
                self._rl_policy.observe(query, action_idx, context, reply)
            except Exception as e:
                logger.warning(f"[RL] observe failed: {e}")
        
        thread = threading.Thread(target=observe, daemon=True)
        thread.start()

    def stream_rag_summarize(self, query: str, enable_sft=True, enable_rl=True):
        """流式版本，支持SFT和RL"""
        # RL选择最优k值
        k = int(chroma_conf["k"])
        action_idx = None
        
        if enable_rl and self._rl_loaded and self._rl_policy:
            try:
                k, action_idx = self._rl_policy.choose_k(query)
            except Exception as e:
                logger.warning(f"[RL] choose_k failed: {e}")
                k = int(chroma_conf["k"])
        
        # 检索文档
        context_docs = self.retrieve_docs(query, k=k)
        
        # 构建上下文
        context = ""
        for i, doc in enumerate(context_docs):
            context += f"资料{i+1}：{doc.page_content}\n"

        # SFT样例注入
        sft_block = ""
        if enable_sft and self._sft_loaded and self._sft_store:
            try:
                sft_block = self._sft_store.format_exemplar_block(query)
            except Exception as e:
                logger.warning(f"[SFT] format_exemplar_block failed: {e}")

        # 构建prompt
        prompt_text = self.prompt_template.format(input=query, context=context, sft_exemplars=sft_block)
        
        # 流式输出
        reply_buffer = ""
        for chunk in self.model.stream(prompt_text):
            content = getattr(chunk, 'content', str(chunk))
            reply_buffer += content
            yield content
        
        # RL奖励更新（后台异步）
        if enable_rl and self._rl_loaded and self._rl_policy and action_idx is not None:
            self._async_rl_observe(query, action_idx, context, reply_buffer)


# for testing
if __name__ == '__main__':
    vs = VectorStoreService()
    rag = RagSummarizeService(vs)

    print(rag.rag_summarize("露天煤矿适合哪种挖掘机？"))
