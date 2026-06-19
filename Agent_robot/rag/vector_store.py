from langchain_core.documents import Document
from model.factory import embed_model
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
import os
from utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, csv_loader, pdf_loader, txt_loader
from utils.logger_handler import logger
from utils.path_tools import get_abs_path


class _SimpleTextSplitter:
    """简单的文本分割器，避免使用有问题的 langchain_text_splitters"""

    def __init__(self, chunk_size=200, chunk_overlap=20, separators=None, length_function=len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", " ", ".", "!", "?", ",", "，"]
        self.length_function = length_function

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        # 先按分隔符分
        pieces = [text]
        for sep in self.separators:
            new_pieces = []
            for p in pieces:
                if not p:
                    continue
                if sep == " ":
                    # 空白按字符切
                    new_pieces.extend([x for x in p.split(sep) if x])
                else:
                    new_pieces.extend([x for x in p.split(sep) if x])
            pieces = new_pieces

        # 再按 chunk_size 合并
        chunks: list[str] = []
        buf = ""
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if not buf:
                buf = p
                continue
            if self.length_function(buf) + 1 + self.length_function(p) <= self.chunk_size:
                buf = buf + "\n" + p
            else:
                chunks.append(buf)
                if self.length_function(p) >= self.chunk_size:
                    # 单段过长，直接切
                    s = 0
                    while s < self.length_function(p):
                        chunks.append(p[s:s + self.chunk_size])
                        s += self.chunk_size - self.chunk_overlap
                    buf = ""
                else:
                    buf = p
        if buf:
            chunks.append(buf)

        # 应用 overlap（简化为：相邻 chunk 头部插入 overlap）
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped: list[str] = []
            prev_tail = ""
            for c in chunks:
                if prev_tail and self.length_function(c) + self.length_function(prev_tail) <= self.chunk_size:
                    overlapped.append(prev_tail + c)
                else:
                    overlapped.append(c)
                tail_len = min(self.chunk_overlap, self.length_function(c))
                prev_tail = c[-tail_len:] if tail_len > 0 else ""
            return overlapped
        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        out: list[Document] = []
        for d in documents:
            for chunk in self.split_text(d.page_content):
                if chunk.strip():
                    out.append(Document(page_content=chunk, metadata=d.metadata))
        return out


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = _SimpleTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self, k: int | None = None):
        kk = chroma_conf["k"] if k is None else k
        return self.vector_store.as_retriever(search_kwargs={"k": kk})


    def load_document(self):

        def check_md5_hex(md5_for_check):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

            return False

        def save_md5_hex(md5_for_save):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save+"\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            elif read_path.endswith("pdf"):
                return pdf_loader(read_path)
            elif read_path.endswith("csv"):
                return csv_loader(read_path)
            else:
                return []

        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]))
        # 从配置文件里指定的data_path目录中，找出所有文件后缀在allow_knowledge_file_type列表里的文件，返回它们的路径列表，存到allowed_files_path里。

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if not md5_hex:  # 处理MD5计算失败的情况
                logger.warning(f"[加载知识库] {path} MD5计算失败，跳过")
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库] {path} 内容已经存在于知识库，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库] {path} 无有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库] {path} 分片后无内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)
                 #将向量存入向量库
                save_md5_hex(md5_hex)#记录这个已经处理好的文件的md5，避免下次重复加载

                logger.info(f"[加载知识库] {path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细报错堆栈，False仅记录报错str
                logger.error(f"[加载知识库] {path} 加载失败：{str(e)}", exc_info=True)
                #exc_info它的作用是在日志中记录完整的异常堆栈信息（Traceback）,如果为false仅记录报错信息本身
                continue


# for testing
if __name__ == '__main__':
    store = VectorStoreService()#实例化前面定义的 VectorStoreService 类，会自动初始化向量数据库（比如 Chroma）、加载配置、初始化日志等。


    store.load_document()  #执行文档加载

    retriever = store.get_retriever()
    # 调用get_retriever()方法，从向量库中获取一个「检索器」对象这个对象的作用是：后续可以通过它，用自然语言查询向量库中的内容

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
