from app.embeddings.embedding_factory import EmbeddingFactory
from app.vectorstores.qdrant_store import QdrantStore

embeddings = EmbeddingFactory.get_embeddings()

store = QdrantStore(embeddings)

print("Successfully connected to Qdrant!")

print()

collections = store.client.get_collections().collections

print("Collections:")

for collection in collections:
    print("-", collection.name)