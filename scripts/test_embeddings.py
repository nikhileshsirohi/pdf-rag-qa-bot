from app.pdf_loader import load_pdf
from app.text_splitter import split_text
from app.embeddings import EmbeddingGenerator

pdf_path = "data/raw_pdfs/sample.pdf"

# Load & chunk
text = load_pdf(pdf_path)
chunks = split_text(text)

print(f"Total chunks: {len(chunks)}")

# Generate embeddings
embedder = EmbeddingGenerator()
embeddings = embedder.embed_texts(chunks)

print("Embeddings shape:", embeddings.shape)
print("Sample embedding (first 5 values):", embeddings[0][:5])
