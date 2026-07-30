from langchain_community.chat_message_histories import ChatMessageHistory

_store: dict[str, ChatMessageHistory] = {}


class MemoryService:

    def get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in _store:
            _store[session_id] = ChatMessageHistory()

        return _store[session_id]

    def add_user_message(self, session_id: str, message: str):
        history = self.get_history(session_id)
        history.add_user_message(message)

    def add_ai_message(self, session_id: str, message: str):
        history = self.get_history(session_id)
        history.add_ai_message(message)