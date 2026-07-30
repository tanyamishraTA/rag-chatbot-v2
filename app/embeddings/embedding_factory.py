# app/embeddings/embedding_factory.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse

from app.core.config import get_settings


class EmbeddingFactory:

    @staticmethod
    def get_dense_embeddings():
        settings = get_settings()

        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    @staticmethod
    def get_sparse_embeddings():
        return FastEmbedSparse(
            model_name="Qdrant/bm25"
        )