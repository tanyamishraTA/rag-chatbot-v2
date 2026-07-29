from app.pipelines.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

question = input("Ask a question: ")

response = pipeline.run(question)

print("\nAnswer\n")
print(response["answer"])

print("\nSources\n")

for source in response["sources"]:
    print(
        f"{source['source']} (Page {source['page']})"
    )