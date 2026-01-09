from app.pdf_loader import load_pdf
from app.text_splitter import split_text
from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever
from app.rag_pipeline import RAGPipeline


pdf_path = "data/raw_pdfs/sample.pdf"

# Load and chunk
text = load_pdf(pdf_path)
chunks = split_text(text)

# Embed chunks
embedder = EmbeddingGenerator()
chunk_embeddings = embedder.embed_texts(chunks).numpy()

# Create FAISS index
retriever = FAISSRetriever(embedding_dim=chunk_embeddings.shape[1])
retriever.add_embeddings(chunk_embeddings, chunks)

# -------- CHOOSE PROVIDER HERE --------
rag = RAGPipeline(
    retriever=retriever,
    provider="huggingface",   # "openai" | "gemini" | "huggingface"
    api_key=None,             # required only for openai/gemini
    model=None                # optional override
)

question = "What is Machine Learning?"
answer = rag.answer_question(question)

print("Question:", question)
print("\nAnswer:\n", answer)
