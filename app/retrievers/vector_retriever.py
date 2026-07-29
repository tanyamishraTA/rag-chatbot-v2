from langchain_core.documents import Document

from app.core.config import get_settings
from app.embeddings.embedding_factory import EmbeddingFactory
from app.vectorstores.qdrant_store import QdrantStore


class VectorRetriever:
    """
    Retrieves the most relevant document chunks from Qdrant.
    """

    def __init__(self):
        settings = get_settings()

        embeddings = EmbeddingFactory.get_embeddings()

        self.vector_store = QdrantStore(embeddings)
        self.top_k = settings.top_k

    def retrieve(self, query: str) -> list[Document]:
        """
        Retrieve the most relevant chunks for a query.
        """
        return self.vector_store.similarity_search(
            query=query,
            k=self.top_k,
        )