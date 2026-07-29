from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveChunker:
    """
    Splits documents into overlapping chunks while preserving useful metadata.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata = {
                "source": Path(chunk.metadata["source"]).name,
                "page": chunk.metadata["page"] + 1,  # Convert to 1-based page numbers
            }

        return chunks