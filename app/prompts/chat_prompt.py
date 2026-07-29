from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant that answers questions ONLY from the provided context.

Rules:
1. Use only the provided context.
2. Do not make up information.
3. If the answer is not present in the context, reply exactly:
   "I don't have enough information to answer this question."
4. Keep answers concise and factual.
""",
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
""",
        ),
    ]
)