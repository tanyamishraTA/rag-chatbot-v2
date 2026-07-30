from app.services.retriever_service import RetrieverService

retriever = RetrieverService()

docs = retriever.retrieve("leave policy")

print(f"Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs, start=1):
    print(f"Result {i}")
    print(doc.metadata)
    print(doc.page_content[:200])
    print("-" * 80)