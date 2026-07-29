from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings


class QdrantStore:
    """
    Wrapper around Qdrant vector database.
    """

    def __init__(self, embeddings: Embeddings):
        settings = get_settings()

        self.collection_name = settings.qdrant_collection

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

        self.embeddings = embeddings

        self._create_collection()

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def _create_collection(self) -> None:
        """
        Create the collection if it doesn't already exist.
        """

        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            embedding_dimension = len(self.embeddings.embed_query("dimension check"))

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(self, documents: list[Document]) -> None:
        """
        Store documents in Qdrant.
        """

        self.vectorstore.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 3,
    ) -> list[Document]:
        """
        Retrieve similar documents.
        """

        return self.vectorstore.similarity_search(
            query=query,
            k=k,
        )