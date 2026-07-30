from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    """

    # Qdrant
    qdrant_url: str = Field(...)
    qdrant_api_key: str | None = None
    qdrant_collection: str = Field(...)

    # Embeddings 
    embedding_model: str

    # Gemini 
    gemini_api_key: str | None = None
    gemini_model: str

    # Ollama 
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # Retrieval 
    top_k: int = 10
    rerank_top_k: int = 3

    # Toggle 
    use_gemini: bool = False
    

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    """
    return Settings()