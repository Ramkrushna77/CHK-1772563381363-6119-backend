# CHK-1772563381363-6119-backend

**Project Overview**
This project is the backend of an AI Mental Health Assistant built using a Hybrid RAG (Retrieval Augmented Generation) system.
The system retrieves information from mental health PDF documents and web sources, then generates answers using a language model.
The goal of this project is to help users get information about mental health topics such as stress, anxiety, and depression.

**Technologies Used**
Python
FAISS (Vector Database)
Sentence Transformers
HuggingFace Transformers
FLAN-T5 Language Model
DuckDuckGo Web Search
PyPDF (PDF text extraction)

**How the System Works**
PDF documents related to mental health are collected.
The text is extracted from PDFs.
The text is split into smaller chunks.
Chunks are converted into embeddings using Sentence Transformers.
These embeddings are stored in a FAISS vector database.
When a user asks a question:
The system finds relevant chunks from the database.
It may also retrieve information from the web.

## 🏗️ Architecture Flow
The system processes user interactions through a multi-step pipeline to provide accurate risk analysis and recommendations.
```text
User
 │
 │  (Face + Voice + Chat)
 ▼
Frontend (React Web App)
 │
 ▼
Backend (FastAPI)
 │
 ├── Facial Emotion Detection (Computer Vision / Webcam)
 ├── Speech Emotion Detection (Audio Processing)
 ├── Chat Sentiment Analysis (NLP)
 ├── User Interaction Tracking
 │
 ▼
Emotion Aggregation Engine
 │
 ▼
Mental Health Risk Analyzer
 │
 ▼
AI Report Generator
 │
 ▼
RAG Chatbot (Hybrid RAG Knowledge Base)
 │
 ▼
Recommendations + Guidance

The combined context is given to the language model.

The language model generates a final answer.
