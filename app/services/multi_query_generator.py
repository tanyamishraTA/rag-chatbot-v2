import re
from app.llms.llm_service import LLMService
from app.prompts.multi_query_prompt import multi_query_prompt


class MultiQueryGenerator:

    def __init__(self):
        self.llm = LLMService()

    def generate_queries(
        self,
        question: str,
        model: str | None = None,
    ) -> list[str]:
        """
        Generates alternative versions of the question and returns a list of queries including the original question.
        Falls back to [question] if LLM query generation fails.
        """
        queries = [question]

        try:
            prompt = multi_query_prompt.invoke({"question": question})

            raw_output = self.llm.generate(
                prompt,
                model=model,
            )

            for line in raw_output.splitlines():
                cleaned = line.strip()
                # Strip potential bullet points or list numbers like "1. ", "- ", etc.
                cleaned = re.sub(r"^(\d+[\.\)]|\-|\*)\s*", "", cleaned).strip()
                if cleaned and cleaned not in queries:
                    queries.append(cleaned)

        except Exception as e:
            print(f"Warning: MultiQueryGenerator failed ({e}). Falling back to single query.")

        return queries
