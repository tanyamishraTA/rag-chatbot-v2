from app.chunking.recursive_chunker import RecursiveChunker
from app.loaders.pdf_loader import PDFLoader
from langchain_core.documents import Document


class DocumentService:
    """
    Responsible for loading and chunking documents.
    """

    def __init__(self, documents_path: str = "documents"):
        self.loader = PDFLoader(documents_path)
        self.chunker = RecursiveChunker()

    def get_documents(self) -> list[Document]:
        return self.loader.load()

    def get_chunks(self) -> list[Document]:
        documents = self.get_documents()
        return self.chunker.split(documents)