from langchain.retrievers import EnsembleRetriever


class HybridRetriever:
    """
    Combines Vector Search and BM25 Search.
    """

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
    ):
        self.retriever = EnsembleRetriever(
            retrievers=[vector_retriever,bm25_retriever,],
            weights=[0.7,0.3],
        )

    def retrieve(self, query: str):
        return self.retriever.invoke(query)