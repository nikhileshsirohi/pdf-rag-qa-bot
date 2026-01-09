from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever


class RAGPipeline:
    """
    This class connects:
    - embeddings
    - FAISS retrieval
    - prompt building
    - LLM generation
    """

    def __init__(self, retriever: FAISSRetriever):
        self.retriever = retriever
        self.embedder = EmbeddingGenerator()

    def build_prompt(self, context_chunks: list[str], question: str) -> str:
        """
        Build prompt for the LLM using retrieved chunks.
        """

        context_text = "\n\n".join(context_chunks)

        prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not present in the context or not relevant to the question, say:
"I could not find the answer in the provided document."

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:
"""
        return prompt.strip()

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """
        Full RAG flow:
        - Embed question
        - Retrieve chunks
        - Build prompt
        - Call LLM
        """

        # Step 1: Embed the user question
        query_embedding = self.embedder.embed_texts([question]).numpy()

        # Step 2: Retrieve top-k relevant chunks
        results = self.retriever.search(query_embedding, top_k=top_k)

        if not results:
            return "No relevant information found."

        context_chunks = [res["text"] for res in results]

        # Step 3: Build prompt
        prompt = self.build_prompt(context_chunks, question)

        # Step 4: Call LLM (mocked here)
        answer = self.call_llm(prompt)

        return answer

    def call_llm(self, prompt: str) -> str:
        """
        This function represents the LLM call.
        Replace this with OpenAI / Gemini / LLaMA later.
        """

        # For now, we just return the prompt end to show flow
        # In real use, this will call an API
        return "[LLM RESPONSE WILL APPEAR HERE]"
