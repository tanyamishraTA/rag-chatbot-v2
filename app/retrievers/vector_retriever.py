from app.core.config import get_settings
from app.embeddings.embedding_factory import EmbeddingFactory
from app.vectorstores.qdrant_store import QdrantStore
from app.rerankers.cross_encoder_reranker import CrossEncoderReranker


class VectorRetriever:

    def __init__(self):
        settings = get_settings()

        dense = EmbeddingFactory.get_dense_embeddings()
        sparse = EmbeddingFactory.get_sparse_embeddings()

        self.vector_store = QdrantStore(
            dense_embeddings=dense,
            sparse_embeddings=sparse,
        )

        self.reranker = CrossEncoderReranker()

        self.top_k = settings.top_k
        self.rerank_top_k = settings.rerank_top_k

    def retrieve(self, query: str):

        documents = self.vector_store.similarity_search(
            query=query,
            k=self.top_k,
        )

        reranked_documents = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=self.rerank_top_k,
        )

        return reranked_documents