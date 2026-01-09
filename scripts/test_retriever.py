from app.pdf_loader import load_pdf
from app.text_splitter import split_text
from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever

pdf_path = "data/raw_pdfs/sample.pdf"

# Load & chunk
text = load_pdf(pdf_path)
chunks = split_text(text)

# Embed
embedder = EmbeddingGenerator()
embeddings = embedder.embed_texts(chunks).numpy()

# Create retriever and add embeddings
retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
retriever.add_embeddings(embeddings, chunks)

# SAVE INDEX
retriever.save()

print("Index saved successfully!")

# Reload retriever (simulate restart)
retriever2 = FAISSRetriever(embedding_dim=embeddings.shape[1])

# Query
query = "What is this document about?"
query_embedding = embedder.embed_texts([query]).numpy()
results = retriever2.search(query_embedding)

print("Retrieved after reload:")
print(results[0]["text"][:300])