from app.loaders.pdf_loader import PDFLoader
from app.chunking.recursive_chunker import RecursiveChunkingService


class DocumentService:

    def __init__(self):
        self.loader = PDFLoader("documents")
        self.chunker = RecursiveChunkingService()

    def get_chunks(self):
        documents = self.loader.load()
        return self.chunker.chunk_documents(documents)