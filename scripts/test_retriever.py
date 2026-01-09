import numpy as np
from app.pdf_loader import load_pdf
from app.text_splitter import split_text
from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever

pdf_path = "data/raw_pdfs/sample.pdf"

# Step 1: Load and chunk PDF
text = load_pdf(pdf_path)
chunks = split_text(text)

print(f"Total chunks: {len(chunks)}")

# Step 2: Generate embeddings
embedder = EmbeddingGenerator()
chunk_embeddings = embedder.embed_texts(chunks).numpy()
print("Chunks & Chunbks embeddings")
print(chunks[0][:50])
print(chunk_embeddings[0][:10])  # Print first 10 values of first embedding
# Step 3: Create FAISS index
retriever = FAISSRetriever(embedding_dim=chunk_embeddings.shape[1])
retriever.add_embeddings(chunk_embeddings, chunks)

# Step 4: Query
query = "What is this document about?"
query_embedding = embedder.embed_texts([query]).numpy()

results = retriever.search(query_embedding, top_k=3)

print("\nTop retrieved chunks:\n")
for i, res in enumerate(results, 1):
    print(f"Result {i} (score={res['score']:.3f})")
    print(res["text"][:300])
    print("-" * 60)
