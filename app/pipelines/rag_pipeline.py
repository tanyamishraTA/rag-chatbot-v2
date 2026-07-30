from app.llms.llm_service import LLMService
from app.memory.memory_service import MemoryService
from app.prompts.chat_prompt import chat_prompt
from app.retrievers.vector_retriever import VectorRetriever
from app.services.query_rewriter import QueryRewriter


class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline.
    """

    def __init__(self):
        self.retriever = VectorRetriever()
        self.llm_service = LLMService()
        self.memory = MemoryService()
        self.query_rewriter = QueryRewriter()

    def run(
        self,
        session_id: str,
        question: str,
        model: str,
    ) -> dict:

        # Load conversation history
        history = self.memory.get_history(session_id)

        history_text = "\n".join(
            f"{message.type}: {message.content}"
            for message in history.messages
        )

        # Rewrite follow-up question only if there is previous conversation
        if history.messages:
            standalone_question = self.query_rewriter.rewrite(
                history=history_text,
                question=question,
                model=model,
            )
        else:
            standalone_question = question

        # Retrieve documents using rewritten question
        documents = self.retriever.retrieve(standalone_question)

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        # Generate final answer
        prompt = chat_prompt.invoke(
            {
                "history": history_text,
                "context": context,
                "question": question,
            }
        )

        answer = self.llm_service.generate(
            prompt=prompt,
            model=model,
        )

        # Save conversation
        self.memory.add_user_message(
            session_id=session_id,
            message=question,
        )

        self.memory.add_ai_message(
            session_id=session_id,
            message=answer,
        )

        sources = [
            {
                "source": doc.metadata["source"],
                "page": doc.metadata["page"],
            }
            for doc in documents
        ]

        return {
            "answer": answer,
            "sources": sources,
        }