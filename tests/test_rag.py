"""
test_rag.py – Unit tests for the RAG complaint intelligence service.

Run with:
    pytest tests/test_rag.py -v

NOTE: These tests mock the heavy LLM and FAISS components so they run
      without requiring a GPU or a downloaded Llama 2 checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class MockBatchEncoding(dict):
    def to(self, device):
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to build mock complaints documents
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_COMPLAINTS = [
    "The billing department charged me twice for the same service in January.",
    "My subscription was renewed without my consent and I was never notified.",
    "Customer service was unhelpful and rude when I called to dispute a charge.",
    "The product stopped working after one week of normal use.",
    "I requested a refund three weeks ago and still have not received it.",
]


@pytest.fixture
def complaints_dir(tmp_path: Path) -> Path:
    """Creates synthetic complaint text files in a temp directory."""
    d = tmp_path / "complaints"
    d.mkdir()
    for i, text in enumerate(SAMPLE_COMPLAINTS):
        (d / f"complaint_{i}.txt").write_text(text, encoding="utf-8")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# RAG Engine – unit tests (mocked LLM + real FAISS)
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGEngine:

    @pytest.fixture(autouse=True)
    def patch_paths_and_llm(self, complaints_dir, tmp_path):
        """
        Redirect document / index paths to tmp dirs and mock the heavy
        LLM components so tests stay fast.
        """
        index_dir = tmp_path / "rag" / "index"
        index_dir.mkdir(parents=True)

        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MockBatchEncoding({"input_ids": MagicMock(shape=[1, 10])})
        mock_tokenizer.decode.return_value = "This is a mocked answer about billing."
        mock_tokenizer.eos_token_id = 2

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = MagicMock()

        # Mock SentenceTransformer embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = lambda sentences, **kwargs: np.array([[1.0] + [0.0]*383 for _ in sentences], dtype=np.float32)

        with (
            patch("services.rag.rag_engine.COMPLAINTS_DIR", complaints_dir),
            patch("services.rag.rag_engine.INDEX_DIR", index_dir),
            patch("services.rag.rag_engine.INDEX_PATH", index_dir / "faiss.index"),
            patch("services.rag.rag_engine.CHUNKS_PATH", index_dir / "chunks.pkl"),
            patch("services.rag.rag_engine.load_llm", return_value=(mock_tokenizer, mock_model)),
            patch("sentence_transformers.SentenceTransformer", return_value=mock_embedder),
        ):
            # Reset the singleton between tests
            import services.rag.rag_engine as eng
            eng.RAGEngine._instance = None
            self.engine = eng.RAGEngine.get()
            yield
            eng.RAGEngine._instance = None

    def test_index_built(self):
        """Index should contain at least as many vectors as complaint chunks."""
        assert self.engine._index.ntotal >= len(SAMPLE_COMPLAINTS)

    def test_retrieve_returns_chunks(self):
        results = self.engine.retrieve("billing charge", top_k=3)
        assert len(results) >= 1
        assert all(hasattr(r, "text") for r in results)

    def test_answer_returns_response(self):
        result = self.engine.answer("What billing complaints exist?", top_k=3)
        assert isinstance(result.answer, str)
        assert len(result.sources) >= 1

    def test_sources_have_required_fields(self):
        result = self.engine.answer("refund issues", top_k=2)
        for source in result.sources:
            assert "chunk_id" in source
            assert "source_file" in source
            assert "snippet" in source

    def test_chunk_text_splits_correctly(self):
        from services.rag.rag_engine import chunk_text
        words = " ".join(["word"] * 1000)
        chunks = chunk_text(words, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # Each chunk should have at most 100 words
        for chunk in chunks:
            assert len(chunk.split()) <= 100


# ─────────────────────────────────────────────────────────────────────────────
# RAG FastAPI endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGEndpoints:

    @pytest.fixture(autouse=True)
    def setup_client(self, complaints_dir, tmp_path):
        index_dir = tmp_path / "rag" / "index"
        index_dir.mkdir(parents=True)

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MockBatchEncoding({"input_ids": MagicMock(shape=[1, 10])})
        mock_tokenizer.decode.return_value = "Mocked answer: billing issues detected."
        mock_tokenizer.eos_token_id = 2

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = MagicMock()

        # Mock SentenceTransformer embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = lambda sentences, **kwargs: np.array([[1.0] + [0.0]*383 for _ in sentences], dtype=np.float32)

        with (
            patch("services.rag.rag_engine.COMPLAINTS_DIR", complaints_dir),
            patch("services.rag.rag_engine.INDEX_DIR", index_dir),
            patch("services.rag.rag_engine.INDEX_PATH", index_dir / "faiss.index"),
            patch("services.rag.rag_engine.CHUNKS_PATH", index_dir / "chunks.pkl"),
            patch("services.rag.rag_engine.load_llm", return_value=(mock_tokenizer, mock_model)),
            patch("sentence_transformers.SentenceTransformer", return_value=mock_embedder),
        ):
            import services.rag.rag_engine as eng
            eng.RAGEngine._instance = None

            import importlib
            import services.rag.main as rag_main
            importlib.reload(rag_main)

            import anyio
            from httpx import AsyncClient, ASGITransport

            class SimpleTestClient:
                def __init__(self, app, base_url: str = "http://testserver") -> None:
                    self._client = AsyncClient(transport=ASGITransport(app=app), base_url=base_url)
                def get(self, url: str, **kwargs):
                    async def _run():
                        return await self._client.get(url, **kwargs)
                    return anyio.run(_run)
                def post(self, url: str, **kwargs):
                    async def _run():
                        return await self._client.post(url, **kwargs)
                    return anyio.run(_run)

            self.client = SimpleTestClient(rag_main.app)
            yield
            eng.RAGEngine._instance = None

    def test_health_check(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_answer_endpoint(self):
        resp = self.client.post("/answer", json={"query": "What are the most common billing complaints?", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_short_query_rejected(self):
        resp = self.client.post("/answer", json={"query": "hi"})
        assert resp.status_code == 422  # Pydantic min_length validation
