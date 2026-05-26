"""
main.py – Complaint Intelligence RAG FastAPI Service
=====================================================
Loads the RAGEngine at startup and serves POST /answer.

Start with:
    uvicorn services.rag.main:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

from services.rag.rag_engine import RAGEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan – warm up the RAG engine
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading RAG engine…")
    try:
        RAGEngine.get()          # builds / loads index and LLM
        log.info("RAG engine ready.")
    except Exception as exc:
        log.error("RAG engine failed to initialise: %s", exc)
    yield
    log.info("RAG service shut down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Complaint Intelligence RAG Service",
    description=(
        "Answers questions about customer complaints using Llama 2 "
        "backed by a FAISS vector store. Returns cited evidence."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class AnswerRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        example="What are the most common reasons customers complain about billing?",
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    groq_api_key: str | None = Field(None, description="Optional Groq API Key to generate answer using Groq cloud")


class SourceItem(BaseModel):
    chunk_id: str
    source_file: str
    snippet: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    query: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"], summary="Health check")
async def health() -> dict:
    try:
        engine = RAGEngine.get()
        index_size = engine._index.ntotal
        status = "ok"
    except Exception as exc:
        index_size = 0
        status = f"degraded: {exc}"
    return {"status": status, "index_vectors": index_size}


@app.post(
    "/answer",
    response_model=AnswerResponse,
    tags=["inference"],
    summary="Answer a complaint‑related question with cited evidence",
)
async def answer(request: AnswerRequest) -> AnswerResponse:
    """
    Retrieve the most relevant complaint passages and generate a
    grounded answer using Llama 2. Each source includes the originating
    file name and a text snippet.
    """
    try:
        engine = RAGEngine.get()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG engine not ready. Ensure complaint documents exist in "
                "data/complaints/ and the service started correctly."
            ),
        ) from exc

    try:
        result = engine.answer(request.query, top_k=request.top_k, groq_api_key=request.groq_api_key)
    except Exception as exc:
        log.exception("RAG inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnswerResponse(
        answer=result.answer,
        sources=[SourceItem(**s) for s in result.sources],
        query=request.query,
    )


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": "complaint-rag", "docs": "/docs"})
