from app.llms.llm_factory import LLMFactory


class LLMService:

    def generate(
        self,
        prompt,
        model: str | None = None,
    ) -> str:

        llm = LLMFactory.get_llm(model)

        response = llm.invoke(prompt)

        return response.content