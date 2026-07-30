from app.retrievers.vector_retriever import VectorRetriever
from app.retrievers.bm25_retriever import BM25DocumentRetriever
from app.retrievers.hybrid_retriever import HybridRetriever
from app.services.document_service import DocumentService


class RetrieverService:
    """
    Builds and manages the application's retrievers.
    """

    def __init__(self):
        document_service = DocumentService()

        chunks = document_service.get_chunks()

        vector_retriever = VectorRetriever().vector_store.as_retriever(
            search_kwargs={"k": 10}
        )

        bm25_retriever = BM25DocumentRetriever(
            documents=chunks,
            top_k=10,
        ).get_retriever()

        self.hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
        )

    def retrieve(self, query: str):
        return self.hybrid.retrieve(query)