from app.llms.llm_service import LLMService
from app.prompts.rewrite_prompt import rewrite_prompt


class QueryRewriter:

    def __init__(self):
        self.llm = LLMService()

    def rewrite(
        self,
        history: str,
        question: str,
        model: str,
    ) -> str:

        prompt = rewrite_prompt.invoke(
            {
                "history": history,
                "question": question,
            }
        )

        return self.llm.generate(
            prompt,
            model=model,
        ).strip()