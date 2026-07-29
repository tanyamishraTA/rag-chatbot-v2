from app.pipelines.rag_pipeline import RAGPipeline


class RAGService:

    def __init__(self):
        self.pipeline = RAGPipeline()

    def chat(self,question: str,model: str,):
        return self.pipeline.run(
            question=question,
            model=model,
            )