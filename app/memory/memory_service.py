from langchain_community.chat_message_histories import ChatMessageHistory
from app.core.config import get_settings


_store: dict[str, ChatMessageHistory] = {}

settings = get_settings()

MAX_HISTORY = settings.history_window


class MemoryService:

    def get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in _store:
            _store[session_id] = ChatMessageHistory()

        return _store[session_id]

    def get_recent_history(self, session_id: str):
        """
        Return only the most recent conversation history.
        """
        history = self.get_history(session_id)
        return history.messages[-MAX_HISTORY:]

    def add_user_message(self, session_id: str, message: str):
        history = self.get_history(session_id)
        history.add_user_message(message)

    def add_ai_message(self, session_id: str, message: str):
        history = self.get_history(session_id)
        history.add_ai_message(message)