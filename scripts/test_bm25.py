from app.services.document_service import DocumentService
from app.retrievers.bm25_retriever import BM25DocumentRetriever

chunks = DocumentService().get_chunks()

retriever = BM25DocumentRetriever(chunks)

docs = retriever.retrieve("leave policy")

print(f"Retrieved {len(docs)} documents\n")

for doc in docs:
    print(doc.metadata)
    print(doc.page_content[:200])
    print("-" * 80)
    