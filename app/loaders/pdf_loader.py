from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    """
    Loads all PDF documents from a directory.
    """

    def __init__(self, documents_path: str):
        self.documents_path = Path(documents_path)

    def load(self) -> list[Document]:
        """
        Load every PDF inside the directory.

        Returns:
            List[Document]
        """
        documents: list[Document] = []

        pdf_files = sorted(self.documents_path.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.documents_path}"
            )

        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()

            documents.extend(docs)

        return documents