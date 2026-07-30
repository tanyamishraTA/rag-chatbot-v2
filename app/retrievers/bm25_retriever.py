from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class BM25DocumentRetriever:
    """
    Wrapper around LangChain BM25Retriever.
    """

    def __init__(
        self,
        documents: list[Document],
        top_k: int = 10,
    ):
        self.retriever = BM25Retriever.from_documents(documents)
        self.retriever.k = top_k

    def get_retriever(self):
        return self.retriever