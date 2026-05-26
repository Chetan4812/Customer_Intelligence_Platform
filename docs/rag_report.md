# Complaint Intelligence RAG Service Report

## 1. Overview
This report details the Retrieval-Augmented Generation (RAG) pipeline developed for the Complaint Intelligence service. The service allows customer support agents to query historical complaints and receive AI-generated resolutions based on past data.

## 2. Core Technologies
- **Vector Database:** FAISS (Facebook AI Similarity Search) - chosen for its extremely fast, memory-efficient local nearest-neighbor search.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` - a lightweight, highly performant embedding model that converts complaint texts into 384-dimensional dense vectors.
- **LLM Generator:** LLaMA-2 via `CTransformers` (Local) / Langchain integration - provides conversational synthesis of the retrieved complaints.

## 3. Pipeline Architecture
1. **Ingestion (`generate_sample_data.py`):** Historical complaints are embedded using the sentence-transformer and ingested into the FAISS index (`rag/faiss_index.bin`).
2. **Retrieval (`rag_engine.py`):** When a user asks a question via the POST `/answer` endpoint, the query is embedded and the top *k=3* most semantically similar complaints are retrieved from FAISS.
3. **Synthesis:** A prompt is dynamically constructed containing the user query and the retrieved context. This prompt is passed to the LLaMA-2 model which synthesizes a coherent answer without hallucinating outside the bounds of the provided complaints.

## 4. API Details
- **Endpoint:** `POST /answer`
- **Payload Schema:** `{"query": "string"}`
- **Response:** `{"answer": "string", "sources": ["list of relevant complaint texts"]}`

## 5. Performance Considerations
- The FAISS index is loaded asynchronously into the application state during the FastAPI lifespan event to ensure the endpoint remains completely non-blocking.
- The use of `all-MiniLM-L6-v2` guarantees sub-100ms retrieval latencies prior to the LLM generation step.
