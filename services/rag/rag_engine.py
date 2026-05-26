"""
rag_engine.py – Complaint Intelligence RAG Engine
==================================================
Builds a FAISS vector index from complaint documents and answers
queries using Llama 2 with cited evidence.

Usage (standalone):
    python services/rag/rag_engine.py --query "What are the most common billing complaints?"
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
COMPLAINTS_DIR = BASE_DIR / "data" / "complaints"
RAG_DIR = BASE_DIR / "rag"
INDEX_DIR = RAG_DIR / "index"
LLM_DIR = RAG_DIR / "llm"

INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"

# ---------------------------------------------------------------------------
# Config (overridable via environment variables)
# ---------------------------------------------------------------------------
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # local path or HF hub id
EMBED_MODEL_NAME: str = os.getenv(
    "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))   # words per chunk
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K: int = int(os.getenv("TOP_K", "5"))
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    """A passage from a complaint document."""
    chunk_id: str
    source_file: str
    text: str
    start_word: int
    end_word: int


@dataclass
class RAGResponse:
    """Structured answer with cited evidence."""
    answer: str
    sources: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_documents(complaints_dir: Path) -> list[tuple[str, str]]:
    """
    Load all .txt / .md / .html files from *complaints_dir*.
    Returns list of (filename, text) tuples.
    """
    docs: list[tuple[str, str]] = []
    supported_suffixes = {".txt", ".md", ".html", ".htm", ".csv"}
    paths = [p for p in complaints_dir.rglob("*") if p.suffix.lower() in supported_suffixes]

    if not paths:
        raise FileNotFoundError(
            f"No complaint documents found in {complaints_dir}. "
            "Add .txt / .md / .html files there before indexing."
        )

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            # Strip HTML tags if present
            if path.suffix.lower() in {".html", ".htm"}:
                text = re.sub(r"<[^>]+>", " ", text)
            docs.append((path.name, text.strip()))
        except Exception as exc:
            log.warning("Could not read %s: %s", path, exc)

    log.info("Loaded %d document(s) from %s", len(docs), complaints_dir)
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        end = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:end]))
        i += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(force_rebuild: bool = False) -> tuple[list[Chunk], object, object]:
    """
    Build (or load) the FAISS index from complaint documents.

    Returns:
        (chunks, faiss_index, embedder)
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load from cache if available and not forced
    if not force_rebuild and INDEX_PATH.exists() and CHUNKS_PATH.exists():
        log.info("Loading existing FAISS index from %s", INDEX_DIR)
        index = faiss.read_index(str(INDEX_PATH))
        with open(CHUNKS_PATH, "rb") as fh:
            chunks: list[Chunk] = pickle.load(fh)
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        log.info("Index loaded: %d vectors", index.ntotal)
        return chunks, index, embedder

    # --- Build from scratch ---
    log.info("Building FAISS index…")
    docs = load_documents(COMPLAINTS_DIR)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    chunks: list[Chunk] = []
    texts_to_embed: list[str] = []

    for filename, doc_text in docs:
        raw_chunks = chunk_text(doc_text)
        for i, chunk_text_val in enumerate(raw_chunks):
            chunk = Chunk(
                chunk_id=f"{filename}::chunk_{i}",
                source_file=filename,
                text=chunk_text_val,
                start_word=i * (CHUNK_SIZE - CHUNK_OVERLAP),
                end_word=i * (CHUNK_SIZE - CHUNK_OVERLAP) + CHUNK_SIZE,
            )
            chunks.append(chunk)
            texts_to_embed.append(chunk_text_val)

    log.info("Encoding %d chunks with '%s'…", len(texts_to_embed), EMBED_MODEL_NAME)
    embeddings = embedder.encode(texts_to_embed, batch_size=32, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner-product (cosine after L2-norm)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Persist
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as fh:
        pickle.dump(chunks, fh)

    log.info("FAISS index built: %d vectors  dim=%d", index.ntotal, dim)
    return chunks, index, embedder


# ---------------------------------------------------------------------------
# LLM loader
# ---------------------------------------------------------------------------

def load_llm():
    """
    Load Llama 2 (or any causal LM) from *LLM_DIR* or HuggingFace Hub.
    Falls back gracefully if the checkpoint is not present.
    """
    if LLM_MODEL_NAME.lower() == "mock":
        log.info("Loading MOCK LLM for local development...")
        class MockBatchEncoding(dict):
            def to(self, device):
                return self
                
        class MockTokenizer:
            def __init__(self):
                self.eos_token_id = 2
            def __call__(self, text, *args, **kwargs):
                import torch
                return MockBatchEncoding({
                    "input_ids": torch.zeros((1, 10), dtype=torch.long),
                    "attention_mask": torch.ones((1, 10), dtype=torch.long)
                })
            def decode(self, tokens, *args, **kwargs):
                return (
                    "Grounded Mock Answer: Based on the retrieved complaints context, "
                    "customers report issues regarding billing overcharges and subscription renewals."
                )

        class MockModel:
            def __init__(self):
                self.device = "cpu"
            def generate(self, *args, **kwargs):
                import torch
                return torch.zeros((1, 20), dtype=torch.long)
            def eval(self):
                pass
                
        return MockTokenizer(), MockModel()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = str(LLM_DIR) if (LLM_DIR / "config.json").exists() else LLM_MODEL_NAME
    log.info("Loading LLM from '%s'…", model_path)

    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",        # uses GPU if available
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
    else:
        log.info("CUDA not available; loading LLM directly to CPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
    model.eval()
    log.info("LLM loaded.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# RAG Engine (stateful singleton)
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Singleton engine that holds the FAISS index and Llama 2 model in memory.
    Call `RAGEngine.get()` to obtain the shared instance.
    """

    _instance: Optional["RAGEngine"] = None

    def __init__(self) -> None:
        self._chunks, self._index, self._embedder = build_index()
        self._tokenizer, self._llm = load_llm()

    @classmethod
    def get(cls) -> "RAGEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[Chunk]:
        """Return the top-k most relevant chunks for *query*."""
        import faiss
        import numpy as np

        q_vec = self._embedder.encode([query])
        faiss.normalize_L2(q_vec)
        scores, indices = self._index.search(q_vec, top_k)
        
        SIMILARITY_THRESHOLD = 0.35
        if len(scores) == 0 or len(scores[0]) == 0 or scores[0][0] < SIMILARITY_THRESHOLD:
            log.warning("RAG Refusal: Top similarity score %.4f is below threshold %.4f", 
                        scores[0][0] if len(scores[0]) else 0, SIMILARITY_THRESHOLD)
            return []
            
        results = [self._chunks[i] for i in indices[0] if i < len(self._chunks)]
        return results

    def answer(self, query: str, top_k: int = TOP_K, groq_api_key: Optional[str] = None) -> RAGResponse:
        """
        Retrieve relevant chunks and generate an answer with citations.

        Returns a RAGResponse with:
            - answer: generated text
            - sources: list of {chunk_id, source_file, snippet}
        """
        import torch

        context_chunks = self.retrieve(query, top_k=top_k)
        if not context_chunks:
            return RAGResponse(
                answer="I'm sorry, but I do not have enough relevant historical complaints to answer this question accurately. Please rephrase or provide more details.",
                sources=[]
            )
            
        context_text = "\n\n".join(
            f"[{c.source_file}] {c.text}" for c in context_chunks
        )

        prompt = (
            "You are a complaint intelligence analyst. "
            "Using ONLY the context below, answer the question concisely. "
            "Cite the source file name whenever you use information from it.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        # Check if Groq API key is provided or present in environment
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if api_key:
            import requests
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": MAX_NEW_TOKENS
            }
            try:
                res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=20)
                if res.status_code == 200:
                    answer_text = res.json()["choices"][0]["message"]["content"].strip()
                else:
                    answer_text = f"Error from Groq API (status code {res.status_code}): {res.text}"
            except Exception as e:
                answer_text = f"Error calling Groq API: {e}"
        else:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._llm.device)

            with torch.no_grad():
                output_ids = self._llm.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            answer_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        sources = [
            {
                "chunk_id": c.chunk_id,
                "source_file": c.source_file,
                "snippet": c.text[:200] + ("…" if len(c.text) > 200 else ""),
            }
            for c in context_chunks
        ]

        return RAGResponse(answer=answer_text, sources=sources)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complaint Intelligence RAG Engine")
    parser.add_argument("--query", type=str, required=True, help="Query to answer")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if args.rebuild_index:
        build_index(force_rebuild=True)

    engine = RAGEngine.get()
    result = engine.answer(args.query, top_k=args.top_k)

    print("\n=== ANSWER ===")
    print(result.answer)
    print("\n=== SOURCES ===")
    for s in result.sources:
        print(f"  [{s['source_file']}]  {s['snippet']}")
