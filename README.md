# ChatGPT Export → Vector DB / RAG Learning Project

An educational project for learning vector databases through a personal ChatGPT RAG system.

## Folder structure

```text
chatgpt-rag/
├── data/
│   ├── raw/
│   ├── processed/
│   └── chunks/
├── embeddings/
├── indexes/
│   ├── faiss_index/
│   └── chroma_db/
├── src/
│   ├── __init__.py
│   ├── parser.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── retriever.py
│   ├── rag.py
│   ├── privacy.py
│   └── backends/
│       ├── __init__.py
│       ├── faiss_backend.py
│       ├── chroma_backend.py
│       └── pinecone_backend.py
├── notebooks/
└── app.py
```

## Phases

- Phase 0 — Scaffold the Project
- Phase 1 — Parse & Understand the Data
- Phase 2 — Clean and Normalize Conversations
- Phase 3 — Chunk Conversation Text
- Phase 4 — Generate Embeddings
- Phase 5 — Build a FAISS Index
- Phase 6 — Explore Chroma and Pinecone
- Phase 7 — Privacy-Minimized Cloud Upload
- Phase 8 — Build the RAG Pipeline
- Phase 9 — Create the Query Application

## Privacy

Raw conversation data stays local at all times. Nothing is sent to Pinecone until an explicit, reviewed minimization step (Phase 7).
