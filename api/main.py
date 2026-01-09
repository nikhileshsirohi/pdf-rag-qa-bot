from fastapi import FastAPI
from pydantic import BaseModel

from app.retriever import FAISSRetriever
from app.rag_pipeline import RAGPipeline

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="PDF RAG QA Bot")

# -----------------------
# Load RAG components ONCE
# -----------------------
retriever = FAISSRetriever(embedding_dim=384)

rag = RAGPipeline(
    retriever=retriever,
    provider="huggingface"   # later: openai / gemini
)

# -----------------------
# Request / Response schemas
# -----------------------
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


# -----------------------
# Routes
# -----------------------

@app.get("/")
def root():
    return {"message": "PDF RAG QA Bot is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    answer = rag.answer_question(request.question)
    return {"answer": answer}
