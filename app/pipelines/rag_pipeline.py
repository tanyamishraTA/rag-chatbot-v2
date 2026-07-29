from app.llms.llm_service import LLMService
from app.prompts.chat_prompt import chat_prompt
from app.retrievers.vector_retriever import VectorRetriever


class RAGPipeline:
    """
    End-to-end Retrieval Augmented Generation pipeline.
    """

    def __init__(self):
        self.retriever = VectorRetriever()
        self.llm_service = LLMService()

    def run(self,question: str,model: str,) -> dict:
        """
        Retrieve relevant context and generate an answer.
        """

        documents = self.retriever.retrieve(question)

        context = "\n\n".join(
            doc.page_content for doc in documents
        )

        prompt = chat_prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        answer = self.llm_service.generate(prompt, model=model,)

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