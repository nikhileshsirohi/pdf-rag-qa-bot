from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever
from app.rag_pipeline import RAGPipeline

def main():
    embedder = EmbeddingGenerator()

    # Load existing FAISS index (NO re-embedding)
    retriever = FAISSRetriever(embedding_dim=384)

    rag = RAGPipeline(
        retriever=retriever,
        provider="huggingface"   # or openai / gemini
    )

    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break

        answer = rag.answer_question(question)
        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
