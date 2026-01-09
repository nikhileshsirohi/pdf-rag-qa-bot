from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever
from app.llm_providers import get_llm_provider

class RAGPipeline:
    """
    This class connects:
    - embeddings
    - FAISS retrieval
    - prompt building
    - LLM generation
    """

    def __init__(
        self,
        retriever: FAISSRetriever,
        provider: str = "huggingface",
        api_key: str | None = None,
        model: str | None = None
    ):
        self.retriever = retriever
        self.embedder = EmbeddingGenerator()

        # Initialize LLM provider dynamically
        self.llm = get_llm_provider(
            provider=provider,
            api_key=api_key,
            model=model
        )

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
        
        for i, r in enumerate(results):
            print(f"Result {i} | Score: {r['score']:.4f}")

        # 🔑 RELEVANCE CHECK
        best_score = results[0]["score"]
        
        if best_score < 0.3:   # threshold (tunable)
            return "The provided document does not contain information related to your question."


        context_chunks = [res["text"] for res in results]

        # Step 3: Build prompt
        prompt = self.build_prompt(context_chunks, question)

        # Step 4: Call LLM (mocked here)
        answer = self.llm.generate(prompt)

        return answer

    # def call_llm(self, prompt: str) -> str:
    #     """
    #     This function represents the LLM call.
    #     Replace this with OpenAI / Gemini / LLaMA later.
    #     """

    #     # For now, we just return the prompt end to show flow
    #     # In real use, this will call an API
    #     return "[LLM RESPONSE WILL APPEAR HERE]"
