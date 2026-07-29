from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings


class EmbeddingFactory:
    """
    Factory responsible for creating embedding models.
    """

    @staticmethod
    def get_embeddings() -> Embeddings:
        settings = get_settings()

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )