from app.pdf_loader import load_pdf
from app.text_splitter import split_text
from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever
from app.rag_pipeline import RAGPipeline


pdf_path = "data/raw_pdfs/sample.pdf"

# Load & chunk PDF
text = load_pdf(pdf_path)
chunks = split_text(text)

# Generate embeddings
embedder = EmbeddingGenerator()
chunk_embeddings = embedder.embed_texts(chunks).numpy()

# Create FAISS retriever
retriever = FAISSRetriever(embedding_dim=chunk_embeddings.shape[1])
retriever.add_embeddings(chunk_embeddings, chunks)

# Create RAG pipeline
rag = RAGPipeline(retriever)

# Ask question
question = "What is this document about?"
answer = rag.answer_question(question)

print("Question:", question)
print("\nAnswer:\n", answer)
