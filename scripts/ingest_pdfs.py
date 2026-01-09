# Run this ONLY when PDFs change
import os
from app.pdf_loader import load_pdf
from app.text_splitter import split_text
from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever

PDF_DIR = "data/raw_pdfs"

def main():
    embedder = EmbeddingGenerator()

    all_chunks = []

    # Load all PDFs
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"Ingesting: {filename}")

            text = load_pdf(pdf_path)
            chunks = split_text(text)
            all_chunks.extend(chunks)

    if not all_chunks:
        print("No PDFs found.")
        return

    # Generate embeddings ONCE
    embeddings = embedder.embed_texts(all_chunks).numpy()

    # Create FAISS index and save
    retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
    retriever.add_embeddings(embeddings, all_chunks)
    retriever.save()

    print("✅ Ingestion complete. FAISS index saved.")


if __name__ == "__main__":
    main()
