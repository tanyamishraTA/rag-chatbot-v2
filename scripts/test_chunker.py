from app.chunking.recursive_chunker import RecursiveChunker
from app.loaders.pdf_loader import PDFLoader

loader = PDFLoader("documents")
documents = loader.load()

print(f"Original pages: {len(documents)}")

chunker = RecursiveChunker(
    chunk_size=500,
    chunk_overlap=100,
)

chunks = chunker.split(documents)

print(f"Chunks created: {len(chunks)}")

print("\nFirst Chunk\n")
print(chunks[0].page_content)

print("\nMetadata\n")
print(chunks[0].metadata)