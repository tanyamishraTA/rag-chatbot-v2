from app.retrievers.vector_retriever import VectorRetriever

retriever = VectorRetriever()

question = "What is the leave policy?"

documents = retriever.retrieve(question)

print(f"\nRetrieved {len(documents)} chunks\n")

for i, doc in enumerate(documents, start=1):
    print("=" * 80)
    print(f"Chunk {i}")
    print("=" * 80)

    print(f"Source : {doc.metadata['source']}")
    print(f"Page   : {doc.metadata['page']}")
    print()
    print(doc.page_content[:500])
    print()