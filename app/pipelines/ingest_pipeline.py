from app.chunking.recursive_chunker import RecursiveChunker
from app.embeddings.embedding_factory import EmbeddingFactory
from app.loaders.pdf_loader import PDFLoader
from app.vectorstores.qdrant_store import QdrantStore


class IngestPipeline:

    def __init__(self, documents_path: str):

        self.loader = PDFLoader(documents_path)
        self.chunker = RecursiveChunker()

        dense = EmbeddingFactory.get_dense_embeddings()
        sparse = EmbeddingFactory.get_sparse_embeddings()

        self.vector_store = QdrantStore(
            dense_embeddings=dense,
            sparse_embeddings=sparse,
        )

    def run(self):

        documents = self.loader.load()

        chunks = self.chunker.split(documents)

        self.vector_store.add_documents(chunks)

        print("Hybrid ingestion completed.")