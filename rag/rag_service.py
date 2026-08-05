from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from rag.reranker import create_reranker
from utils.prompt_loader import load_rag_prompts
from utils.config_handler import chroma_conf
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.logger_handler import logger


def print_prompt(prompt):
    print("=" * 20)
    print(prompt.to_string())
    print("=" * 20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.reranker = create_reranker(
            enabled=chroma_conf.get("reranker_enabled", False),
            model_name=chroma_conf.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            top_k=chroma_conf.get("reranker_top_k", 3),
        )
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    @staticmethod
    def _format_citation(metadata: dict) -> str:
        source_file = metadata.get("source_file", "未知来源")
        chunk_index = metadata.get("chunk_index", -1)
        page = metadata.get("page", -1)

        if page and page >= 0:
            return f"{source_file}，第{page}页，第{chunk_index}段"
        return f"{source_file}，第{chunk_index}段"

    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)

        if not context_docs:
            logger.warning({"event": "rag_no_results", "query": query})
            return "抱歉，我未检索到与您问题相关的可靠资料，无法给出准确回答。建议您：1）换一种方式描述问题；2）咨询人工客服获取帮助。"

        context_docs = self.reranker.rerank(query, context_docs)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            citation = self._format_citation(doc.metadata)
            context += f"【参考资料{counter}】来源：{citation}\n内容：{doc.page_content}\n\n"

        logger.info({
            "event": "rag_summarize",
            "query": query,
            "doc_count": len(context_docs),
            "sources": [doc.metadata.get("source_file", "未知") for doc in context_docs],
        })

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))