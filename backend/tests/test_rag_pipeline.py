"""Automated unit and integration tests for the Staff RAG Engineering Platform."""

import time
from fastapi.testclient import TestClient

from main import app
from app.evaluation.benchmark import run_ablation_benchmark
from app.ingestion.chunking import ChunkingEngine, compute_content_hash
from app.ingestion.pipeline import ingestion_pipeline
from app.models import DocumentInput, EvaluationRequest, QueryInput
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_parent_child_chunking_and_idempotent_dedup():
    """Verifies Parent-Child small-to-big chunking, contextual headers, and SHA-256 deduplication."""
    retrieval_engine.clear()
    engine = ChunkingEngine(parent_chunk_size=600, parent_chunk_overlap=100, child_chunk_size=200, child_chunk_overlap=30)

    sample_md = """# Distributed Consensus Engine
Raft consensus guarantees linearizable state machine replication across quorum nodes within 180ms.

## Error Handling
When ERR_LEDGER_9042 occurs, operators must run aegis-ctl raft resync immediately to restore quorum."""

    chunks_first = engine.chunk_document(
        content=sample_md,
        title="Consensus Spec",
        source="internal://spec.md",
        strategy="parent_child",
    )
    assert len(chunks_first) >= 2
    for chk in chunks_first:
        assert "[Document: Consensus Spec | Source: internal://spec.md" in chk.content
        assert len(chk.parent_content) >= len(chk.content)

    # First ingest adds all chunks; second identical ingest skips 100% as duplicates
    res1 = ingestion_pipeline.ingest(
        DocumentInput(
            content=sample_md,
            title="Consensus Spec",
            source="internal://spec.md",
            chunking_strategy="parent_child",
        )
    )
    assert res1["chunks_indexed_new"] == len(chunks_first)
    assert res1["chunks_skipped_duplicate"] == 0

    res2 = ingestion_pipeline.ingest(
        DocumentInput(
            content=sample_md,
            title="Consensus Spec",
            source="internal://spec.md",
            chunking_strategy="parent_child",
        )
    )
    assert res2["chunks_indexed_new"] == 0
    assert res2["chunks_skipped_duplicate"] == len(chunks_first)


def test_hybrid_retrieval_and_reranking_modes():
    """Verifies Dense, Sparse (BM25), Hybrid RRF, and Cross-Encoder Reranking modes."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    query = "What triggers ERR_LEDGER_9042 and what is the remediation command?"
    for mode in ("dense", "sparse", "hybrid_rrf", "hybrid_rerank"):
        results = retrieval_engine.retrieve(queries=[query], mode=mode, top_k=3)  # type: ignore[arg-type]
        assert len(results) > 0
        top_text = results[0].parent_content.lower()
        if mode in ("sparse", "hybrid_rrf", "hybrid_rerank"):
            assert "err_ledger_9042" in top_text
            assert "aegis-ctl" in top_text


def test_ablation_evaluation_suite():
    """Verifies that the Golden QA Ablation Benchmark computes IR metrics across all 4 modes."""
    retrieval_engine.clear()
    report = run_ablation_benchmark(
        EvaluationRequest(
            top_k=3,
            modes=["dense", "sparse", "hybrid_rrf", "hybrid_rerank"],
            seed_golden_corpus=True,
        )
    )
    assert report["status"] == "success"
    summaries = {row["mode"]: row for row in report["summaries"]}
    assert "hybrid_rerank" in summaries
    assert summaries["hybrid_rerank"]["hit_rate_at_k"] >= 0.8
    assert summaries["hybrid_rerank"]["mrr_at_k"] >= 0.8
    assert "NDCG@3" in report["markdown_report"]


def test_fastapi_endpoints_and_langgraph_job_lifecycle():
    """Verifies /health, /ingest-sample, /retrieve, /submit-job, and /get-job/{job_id}."""
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    seed = client.post("/ingest-sample?strategy=parent_child")
    assert seed.status_code == 200
    assert seed.json()["status"] == "success"

    ret = client.post(
        "/retrieve",
        json={
            "query": "Which PagerDuty rotation ID handles sovereign regulatory inquiries in EMEA?",
            "retrieval_mode": "hybrid_rerank",
            "top_k": 3,
        },
    )
    assert ret.status_code == 200
    assert ret.json()["count"] >= 1
    assert "pd-emea-8821" in ret.json()["results"][0]["parent_content"].lower()

    # Submit async job (uses automatic local thread fallback when RabbitMQ container isn't running)
    submit = client.post(
        "/submit-job",
        json={
            "text": "What triggers ERR_LEDGER_9042 and what is the remediation command?",
            "retrieval_mode": "hybrid_rerank",
            "top_k": 3,
            "enable_query_expansion": True,
            "enable_web_search": False,
            "enable_self_rag": True,
        },
    )
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    # Poll until completed
    final_job = None
    for _ in range(30):
        poll = client.get(f"/get-job/{job_id}")
        assert poll.status_code == 200
        data = poll.json()
        if data["status"] in ("completed", "failed"):
            final_job = data
            break
        time.sleep(0.1)

    assert final_job is not None
    assert final_job["status"] == "completed"
    assert "ERR_LEDGER_9042" in final_job["result"]
    assert final_job["trace"]["crag_decision"] == "sufficient"
    assert final_job["trace"]["faithfulness_score"] > 0.5
