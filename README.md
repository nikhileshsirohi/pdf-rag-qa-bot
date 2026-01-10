# 📄 PDF RAG QA Bot

A **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents once and then ask **unlimited questions** about their content using semantic search and large language models.

This project demonstrates how to build a **local, reusable, and scalable RAG pipeline** instead of sending entire PDFs as prompts to LLMs.

---

## ✨ Features

- 📤 Upload PDF documents (single or multiple over time)
- ✂️ Automatic text extraction and chunking
- 🧠 Semantic search using embeddings
- ⚡ Fast similarity search using **FAISS (Facebook AI Similarity Search)**
- 🤖 Answer generation using multiple LLM providers
- ♻️ Persistent vector store (reuse indexed PDFs without re-uploading)
- 🌐 Simple browser-based UI

---

## 🧱 Tech Stack

- **Python**
- **FastAPI** – backend API
- **FAISS (Facebook AI Similarity Search)** – vector indexing & search
- **Hugging Face Transformers**
- **Sentence Transformers**
- **OpenAI / Gemini (optional via API key)**
- **HTML + JavaScript** – lightweight UI

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/nikhileshsirohi/pdf-rag-qa-bot.git
cd pdf-rag-qa-bot
pip install -r requirements.txt
```
***Start the backend server***
```bash
uvicorn api.main:app --reload
```
***FastAPI docs:***
```bash
http://127.0.0.1:8000/docs
```
**Start the UI**
***Open the UI file***
ui/index.html

## What is this project about?

This project allows you to ask questions from any PDF document you upload.

You can:
- ***Upload a PDF only once***
- ***Ask unlimited questions anytime***
- ***Add more PDFs later (index grows incrementally)***
- ***Reuse the same indexed knowledge every time***

There is no need to upload the same file again.

⸻

### Why not upload PDFs directly to ChatGPT or Gemini?

Uploading PDFs directly to an LLM is not scalable:
- ❌ Prompt size limitations
- ❌ Entire document must be sent every time
- ❌ Expensive and inefficient
- ❌ No persistent memory

### How this project is different
- ***PDFs are not passed as prompts***
- ***Documents are converted into vector embeddings***
- ***Only relevant chunks are retrieved***
- ***Context size stays small and efficient***
- ***Knowledge is persistent and reusable***

⸻

## How it works (RAG Pipeline)

### 1️⃣ PDF Ingestion
- ***User uploads a PDF***
- ***Text is extracted from the document***
- ***Text is split into semantic chunks (paragraph-sized)***

⸻

### 2️⃣ Embedding Generation

Each chunk is converted into a vector embedding using: ***sentence-transformers/all-MiniLM-L6-v2***

#### What are embeddings?
- Dense numerical representations of text
- Semantically similar text → vectors closer in space
- Generated using transformer internals:
- input_ids – token IDs
- - attention_mask – mask for valid tokens
- - Encoder hidden layers
- - Output: fixed-size vector (384 dimensions)

### 3️⃣ Vector Storage using FAISS

FAISS (Facebook AI Similarity Search) is used to store and search embeddings.
	•	Each embedding is assigned an index automatically
	•	Corresponding text chunks are stored at the same index
	•	Cosine similarity is used for semantic search
	•	FAISS index is persisted to disk

This allows reuse across application restarts.

### 4️⃣ Question Answering (Search + Generation)

When a user asks a question:
	1.	Question is converted into an embedding
	2.	FAISS retrieves top-K most similar embeddings
	3.	Matching chunk indices are mapped back to text
	4.	Relevant chunks are passed to an LLM
	5.	LLM refines the context and generates the final answer

## Supported LLMs

### Free (Default)
	•	Hugging Face
	•	google/flan-t5-base
	•	No API key required

⸻

### Optional (Requires API Key)

**Google Gemini**
	•	models/gemini-flash-lite-latest
	•	models/gemini-flash-latest
	•	models/gemini-pro-latest

**OpenAI**
	•	gpt-4o-mini
	•	gpt-4o

Users can select the model from the UI.
If no API key is provided, the system automatically falls back to the free Hugging Face model.

### Persistent Knowledge Base
	•	PDFs are uploaded only once
	•	FAISS index and text metadata are saved locally
	•	Knowledge is reused across sessions
	•	No repeated uploads required


## Author

### Nikhilesh Sirohi
**GitHub: https://github.com/nikhileshsirohi**