from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class BM25DocumentRetriever:
    def __init__(self, documents: list[Document], top_k: int = 5):
        self.retriever = BM25Retriever.from_documents(documents)
        self.retriever.k = top_k

    def retrieve(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)