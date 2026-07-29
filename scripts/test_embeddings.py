from app.embeddings.embedding_factory import EmbeddingFactory

embeddings = EmbeddingFactory.get_embeddings()

vector = embeddings.embed_query(
    "What is the leave policy?"
)

print(f"Vector Dimension : {len(vector)}")

print()

print(vector[:10])
