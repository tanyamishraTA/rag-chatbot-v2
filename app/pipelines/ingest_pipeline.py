from app.chunking.recursive_chunker import RecursiveChunker
from app.embeddings.embedding_factory import EmbeddingFactory
from app.loaders.pdf_loader import PDFLoader
from app.vectorstores.qdrant_store import QdrantStore


class IngestPipeline:
    """
    End-to-end document ingestion pipeline.
    """

    def __init__(self, documents_path: str):
        self.loader = PDFLoader(documents_path)
        self.chunker = RecursiveChunker()

        embeddings = EmbeddingFactory.get_embeddings()

        self.vector_store = QdrantStore(embeddings)

    def run(self) -> None:
        """
        Load → Chunk → Store documents.
        """

        print("Loading documents...")

        documents = self.loader.load()

        print(f"Loaded {len(documents)} pages")

        print("Chunking documents...")

        chunks = self.chunker.split(documents)

        print(f"Created {len(chunks)} chunks")

        print("Uploading to Qdrant...")

        self.vector_store.add_documents(chunks)

        print("Ingestion completed successfully.")