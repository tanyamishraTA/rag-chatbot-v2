from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import get_settings


class QdrantStore:

    def __init__(
        self,
        dense_embeddings: Embeddings,
        sparse_embeddings: FastEmbedSparse,
    ):
        settings = get_settings()

        self.collection_name = settings.qdrant_collection

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        self.dense_embeddings = dense_embeddings
        self.sparse_embeddings = sparse_embeddings

        self._create_collection()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )

    def _create_collection(self):

        collections = self.client.get_collections().collections

        names = [c.name for c in collections]

        if self.collection_name not in names:

            dimension = len(
                self.dense_embeddings.embed_query(
                    "dimension check"
                )
            )

            self.client.create_collection(
                collection_name=self.collection_name,

                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),

                sparse_vectors_config={
                    "langchain-sparse": SparseVectorParams()
                },
            )

    def add_documents(
        self,
        documents: list[Document],
    ):

        self.vectorstore.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 10,
    ) -> list[Document]:

        return self.vectorstore.similarity_search(
            query=query,
            k=k,
        )