import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.pdf_loader import load_pdf
from app.text_splitter import split_text
from app.embeddings import EmbeddingGenerator
from app.retriever import FAISSRetriever
from app.rag_pipeline import RAGPipeline

PDF_DIR = "data/raw_pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

app = FastAPI(title="PDF RAG QA Bot")

# Load components ONCE
retriever = FAISSRetriever(embedding_dim=384)
embedder = EmbeddingGenerator()
rag = RAGPipeline(retriever=retriever, provider="huggingface")

class QuestionRequest(BaseModel):
    question: str

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
    answer = rag.answer_question(request.question)
    return {"answer": answer}

@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    save_path = os.path.join(PDF_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest the uploaded PDF (append to index)
    text = load_pdf(save_path)
    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in PDF")

    embeddings = embedder.embed_texts(chunks).numpy()
    retriever.add_embeddings(embeddings, chunks)
    retriever.save()

    return {
        "message": "PDF uploaded and indexed successfully",
        "chunks_added": len(chunks)
    }
