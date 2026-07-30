from app.pipelines.rag_pipeline import RAGPipeline


class RAGService:

    def __init__(self):
        self.pipeline = RAGPipeline()

    def chat(
        self,
        session_id: str,
        question: str,
        model: str,
    ):
        return self.pipeline.run(
            session_id=session_id,
            question=question,
            model=model,
        )