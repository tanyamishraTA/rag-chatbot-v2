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

    def retrieve_multi(self, queries: list[str], original_query: str):
        """
        Retrieves candidate documents for multiple query variations,
        deduplicates them, and reranks the candidate pool against original_query.
        """
        seen = set()
        deduped_documents = []

        for q in queries:
            docs = self.vector_store.similarity_search(
                query=q,
                k=self.top_k,
            )
            for doc in docs:
                doc_key = (
                    doc.page_content,
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                )
                if doc_key not in seen:
                    seen.add(doc_key)
                    deduped_documents.append(doc)

        reranked_documents = self.reranker.rerank(
            query=original_query,
            documents=deduped_documents,
            top_k=self.rerank_top_k,
        )

        return reranked_documents