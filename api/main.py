import os
import shutil
import pickle
import numpy as np
import faiss

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pdf_loader import load_pdf
from app.text_splitter import split_text
from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever
from app.rag_pipeline import RAGPipeline
from app.llm_providers import get_llm_provider

# ✅ CREATE APP ONCE
app = FastAPI(title="PDF RAG QA Bot")

# ✅ ADD CORS ON THIS APP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ok for local dev
    allow_credentials=False,    # IMPORTANT with "*" (recommended)
    allow_methods=["*"],        # includes OPTIONS + POST
    allow_headers=["*"],
)

PDF_DIR = "data/raw_pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

# Load components ONCE
retriever = FAISSRetriever(embedding_dim=384)
embedder = EmbeddingGenerator()
rag = RAGPipeline(retriever=retriever, provider="huggingface")

class QuestionRequest(BaseModel):
    question: str
    provider: str = "huggingface"
    api_key: str | None = None
    model: str | None = None

class AnswerResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF RAG QA Bot API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    print(f"Received question: {request.question} using provider: {request.provider}")
    llm = get_llm_provider(
        provider=request.provider,
        api_key=request.api_key,
        model=request.model
    )
    rag.llm = llm
    answer = rag.answer_question(request.question)
    return {"answer": answer}

@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    save_path = os.path.join(PDF_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = load_pdf(save_path)
    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    embeddings = embedder.embed_texts(chunks).numpy()
    retriever.add_embeddings(embeddings, chunks)
    retriever.save()

    return {"message": "PDF uploaded and indexed successfully", "chunks_added": len(chunks)}