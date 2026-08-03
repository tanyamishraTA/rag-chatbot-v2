from app.retrievers.vector_retriever import VectorRetriever
from app.services.multi_query_generator import MultiQueryGenerator


def main():
    generator = MultiQueryGenerator()
    retriever = VectorRetriever()

    question = "What is the leave policy?"

    print(f"\nOriginal Question: {question}")

    try:
        queries = generator.generate_queries(question)
        print("\nGenerated Multi-Query Variations:")
        for i, q in enumerate(queries, start=1):
            print(f"  {i}. {q}")

        documents = retriever.retrieve_multi(queries=queries, original_query=question)

        print(f"\nRetrieved {len(documents)} deduplicated & reranked chunks:\n")

        for i, doc in enumerate(documents, start=1):
            print("=" * 80)
            print(f"Reranked Chunk {i}")
            print("=" * 80)
            print(f"Source : {doc.metadata.get('source', 'Unknown')}")
            print(f"Page   : {doc.metadata.get('page', 'Unknown')}")
            print()
            print(doc.page_content[:300])
            if len(doc.page_content) > 300:
                print("...")
            print()

    except Exception as e:
        print(f"Error during MQR test: {e}")


if __name__ == "__main__":
    main()
