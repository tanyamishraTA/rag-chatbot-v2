from app.retrievers.vector_retriever import VectorRetriever


def main():
    retriever = VectorRetriever()

    question = "What is the leave policy?"

    documents = retriever.retrieve(question)

    print(f"\nRetrieved {len(documents)} reranked chunks\n")

    for i, doc in enumerate(documents, start=1):
        print("=" * 80)
        print(f"Reranked Chunk {i}")
        print("=" * 80)

        print(f"Source : {doc.metadata.get('source', 'Unknown')}")
        print(f"Page   : {doc.metadata.get('page', 'Unknown')}")
        print()

        print(doc.page_content[:500])

        if len(doc.page_content) > 500:
            print("...")

        print()


if __name__ == "__main__":
    main()