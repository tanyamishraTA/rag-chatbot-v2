from app.loaders.pdf_loader import PDFLoader

loader = PDFLoader("documents")

documents = loader.load()

print(f"Loaded {len(documents)} pages\n")

doc = documents[0]

print("Page Content:\n")
print(doc.page_content[:500])

print("\nMetadata:\n")
print(doc.metadata)