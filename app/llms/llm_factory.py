from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from app.core.config import get_settings


class LLMFactory:

    @staticmethod
    def get_llm(model: str) -> BaseChatModel:

        settings = get_settings()

        if model == "gemini":
            return ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.gemini_api_key,
                temperature=0,
            )

        if model == "ollama":
            return ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )

        raise ValueError(f"Unsupported model: {model}")