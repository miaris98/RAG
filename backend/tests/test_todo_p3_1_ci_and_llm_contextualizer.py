"""TDD tests for P3.1: CI/Docker dependency parity (python-multipart/requests) & optional Ollama LLM contextualizer."""

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from app.ingestion.chunking import generate_contextual_prefix
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ci_workflow_and_backend_requirements_include_pdf_multipart_dependencies():
    """Verifies backend/requirements.txt and .github/workflows/edd.yml declare python-multipart & requests."""
    req_text = (REPO_ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8").lower()
    edd_text = (REPO_ROOT / ".github" / "workflows" / "edd.yml").read_text(encoding="utf-8").lower()

    assert "python-multipart" in req_text
    assert "requests" in req_text
    assert "python-multipart" in edd_text
    assert "requests" in edd_text


def test_optional_ollama_llm_contextualizer_uses_llm_when_available_and_falls_back_cleanly():
    """P3.1 Anthropic Contextualizer: calls Ollama when enabled+reachable and falls back to extractive prefix offline."""
    doc = "# Atlas Ledger\n\nAtlas settles cross-border euro wires in 90ms.\n\n## Limits\nIt caps batches at 500 items."

    # 1. Mocked live Ollama LLM response
    class _FakeResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "This Limits section defines batch caps for the Atlas euro settlement ledger."
                        }
                    }
                ]
            }

    with patch("app.ingestion.chunking.requests.post", return_value=_FakeResp()):
        prefix_llm, backend_llm = generate_contextual_prefix(
            full_doc_text=doc,
            title="Atlas Spec",
            section="Limits",
            chunk_text="It caps batches at 500 items.",
            use_llm=True,
            return_backend=True,
        )
    assert "[Context:" in prefix_llm
    assert "Atlas euro settlement ledger" in prefix_llm
    assert backend_llm.startswith("ollama_llm")

    # 2. Offline / error fallback -> deterministic extractive prefix without raising
    with patch("app.ingestion.chunking.requests.post", side_effect=ConnectionError("Ollama offline")):
        prefix_fb, backend_fb = generate_contextual_prefix(
            full_doc_text=doc,
            title="Atlas Spec",
            section="Limits",
            chunk_text="It caps batches at 500 items.",
            use_llm=True,
            return_backend=True,
        )
    assert "[Context:" in prefix_fb
    assert "Atlas Spec" in prefix_fb
    assert backend_fb == "extractive_fallback"


def test_ingest_endpoint_accepts_use_llm_contextualizer_flag():
    """POST /ingest accepts use_llm_contextualizer and returns contextualizer_backend telemetry."""
    retrieval_engine.clear()
    with patch("app.ingestion.chunking.requests.post", side_effect=ConnectionError("offline")):
        resp = client.post(
            "/ingest",
            json={
                "title": "Atlas Fallback Spec",
                "content": "# Atlas\n\nAtlas settles cross-border wires.\n\n## Limits\nMax 500 items per batch.",
                "enable_contextual_enrichment": True,
                "use_llm_contextualizer": True,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["contextual_enrichment_enabled"] is True
    assert data["contextualizer_backend"] == "extractive_fallback"
