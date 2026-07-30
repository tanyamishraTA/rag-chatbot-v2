from langchain_core.prompts import ChatPromptTemplate

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Rewrite the user's latest question into a standalone question.

Use the conversation history only to resolve references like:
- it
- they
- this
- that
- he
- she

Do NOT answer the question.
Return only the rewritten question.
""",
        ),
        (
            "human",
            """
Conversation History:
{history}

Latest Question:
{question}
""",
        ),
    ]
)