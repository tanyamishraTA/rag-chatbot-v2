# RAG Chatbot – Intelligent Document Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** chatbot that enables users to ask natural language questions about organizational documents and receive accurate, context-aware responses with source citations.

The chatbot leverages **Hybrid Search (Dense + Sparse Retrieval)**, **Cross-Encoder Re-ranking**, and **Conversational Memory** to deliver highly relevant and context-aware answers.

---

## Features

### Document Ingestion
- PDF document loading
- Recursive document chunking
- Dense embedding generation
- Automatic indexing into Qdrant

### Hybrid Search
- Native Qdrant Hybrid Search
- Dense semantic retrieval
- Sparse BM25 retrieval using FastEmbed
- Top-k document retrieval

### Re-ranking
- Cross-Encoder Re-ranking (`BAAI/bge-reranker-base`)
- Improves retrieval relevance before passing context to the LLM

### Conversational Memory
- Session-based chat history
- Multi-turn conversations
- Query rewriting for follow-up questions
- Configurable history trimming

### Multiple LLM Support
- Ollama
- Google Gemini
- Runtime model selection

### Source Citations
- Returns document source
- Displays page number for every retrieved answer

---

# Tech Stack

## Backend
- FastAPI
- Python

## AI Framework
- LangChain

## Vector Database
- Qdrant Cloud

## Embeddings
- HuggingFace Embeddings
- FastEmbed Sparse (BM25)

## Retrieval
- Native Qdrant Hybrid Search

## Re-ranking
- Cross Encoder (`BAAI/bge-reranker-base`)

## Large Language Models
- Ollama
- Google Gemini

## Frontend
- React
- Tailwind CSS

---

# Project Structure

```text
chatbot-v2
│
├── app
│   ├── api
│   ├── chunking
│   ├── core
│   ├── embeddings
│   ├── llms
│   ├── loaders
│   ├── memory
│   ├── pipelines
│   ├── prompts
│   ├── rerankers
│   ├── retrievers
│   ├── schemas
│   ├── services
│   └── vectorstores
│
├── documents
├── scripts
├── requirements.txt
└── README.md
```

---

# Architecture

## Document Ingestion Pipeline

```text
PDF Documents
      │
      ▼
Recursive Chunking
      │
      ▼
Dense + Sparse Embeddings
      │
      ▼
Qdrant Hybrid Index
```

---

## Retrieval Pipeline

```text
User Question
      │
      ▼
Hybrid Search (Dense + Sparse)
      │
      ▼
Cross Encoder Re-ranking
      │
      ▼
Top Relevant Context
      │
      ▼
LLM
      │
      ▼
Answer + Sources
```

---

## Conversational Pipeline

```text
User Question
      │
      ▼
Load Recent Chat History
      │
      ▼
History Trimming
      │
      ▼
Query Rewriter
      │
      ▼
Hybrid Retrieval
      │
      ▼
Cross Encoder Re-ranking
      │
      ▼
Prompt Construction
      │
      ▼
LLM
      │
      ▼
Store Conversation
      │
      ▼
Response
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/tanyamishraTA/rag-chatbot-v2

cd chatbot-v2
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

```env
# Qdrant
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=chatbot-hybrid

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

# Ollama
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434

# Retrieval
TOP_K=10
RERANK_TOP_K=3

# Memory
HISTORY_WINDOW=6
```

---

# Running the Project

## Start Ollama

```bash
ollama serve
```

---

## Pull the Model

```bash
ollama pull llama3.2
```

---

## Ingest Documents

```bash
python -m scripts.ingest
```

---

## Start Backend

```bash
uvicorn app.main:app --reload
```

---

## Start Frontend

```bash
npm install

npm run dev
```

---

# API

## Chat Endpoint

```
POST /chat
```

### Request

```json
{
  "session_id": "user123",
  "question": "What is the leave policy?",
  "model": "ollama"
}
```

---

### Response

```json
{
  "answer": "Employees are entitled to...",
  "sources": [
    {
      "source": "Trainee - HR Manual.pdf",
      "page": 19
    }
  ]
}
```

---

# Features Implemented

- PDF Loader
- Recursive Chunking
- Dense Embeddings
- Native Qdrant Hybrid Search
- Sparse BM25 Retrieval
- Cross Encoder Re-ranking
- Session-based Conversational Memory
- Query Rewriting
- History Trimming
- Multi-LLM Support (Gemini & Ollama)
- Source Citations
- React Frontend Integration

---

# Future Enhancements

- Redis-based Persistent Memory
- Streaming Responses
- Authentication & User Management
- Multiple Document Collections
- Conversation Summarization
- Feedback & Rating System
- Docker Support
- Docker Compose
- Kubernetes Deployment
- CI/CD Pipeline
- LangSmith Observability

---

# License

This project is intended for educational and learning purposes.

---

# Author 
Tanya Mishra