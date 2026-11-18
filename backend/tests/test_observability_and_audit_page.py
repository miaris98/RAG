"""TDD Test Suite for End-to-End Observability (Langfuse Tracing, Prometheus) & Compliance Audit Page."""

from fastapi.testclient import TestClient

from main import app
from app.agent.graph import execute_rag_pipeline
from app.guardrails.audit import audit_logger
from app.ingestion.pipeline import ingestion_pipeline
from app.models import QueryInput
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.semantic_cache import semantic_cache


client = TestClient(app)


def test_prometheus_metrics_exposition():
    """Step 1.1: GET /metrics/prometheus returns valid Prometheus text exposition metrics."""
    retrieval_engine.clear()
    semantic_cache.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    execute_rag_pipeline(
        QueryInput(
            text="What triggers ERR_LEDGER_9042?",
            enable_web_search=False,
            enable_semantic_cache=True,
        )
    )

    resp = client.get("/metrics/prometheus")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")
    body = resp.text

    expected_metrics = [
        "rag_queries_total",
        "rag_cache_hits_total",
        "rag_crag_fallbacks_total",
        "rag_guardrail_triggers_total",
        "rag_gdpr_erasures_total",
        "rag_latency_ms_p50",
        "rag_latency_ms_p95",
        "rag_latency_ms_p99",
        "rag_corpus_documents_total",
        "rag_corpus_chunks_total",
        "rag_audit_chain_valid",
        'rag_circuit_breaker_open{service="ollama_llm"}',
    ]
    for metric in expected_metrics:
        assert metric in body, f"Missing Prometheus metric: {metric}"


def test_langfuse_compatible_distributed_tracing():
    """Step 1.2: Executing a query records a structured trace + spans retrievable via /traces and /traces/{trace_id}."""
    retrieval_engine.clear()
    semantic_cache.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    out = execute_rag_pipeline(
        QueryInput(
            text="What triggers ERR_LEDGER_9042?",
            enable_web_search=False,
            enable_semantic_cache=False,
        )
    )
    trace_id = out["trace"].get("trace_id")
    assert trace_id is not None and len(trace_id) > 8

    traces_resp = client.get("/traces")
    assert traces_resp.status_code == 200
    traces_data = traces_resp.json()
    assert traces_data["total_traces"] >= 1

    match = [t for t in traces_data["traces"] if t["trace_id"] == trace_id]
    assert len(match) == 1
    t_rec = match[0]
    assert t_rec["name"] == "rag_query_pipeline"
    assert t_rec["tenant_id"] == "default"
    assert "user_roles" in t_rec
    assert t_rec["query_route"] == "lexical_lookup"
    assert isinstance(t_rec["spans"], list) and len(t_rec["spans"]) >= 1
    assert "langfuse_status" in t_rec

    single_resp = client.get(f"/traces/{trace_id}")
    assert single_resp.status_code == 200
    assert single_resp.json()["trace_id"] == trace_id


def test_document_inventory_and_compliance_audit_report():
    """Step 1.3: GET /documents lists indexed docs, DELETE /documents/{doc_id} removes them,
    and GET /compliance/audit-report produces the complete audit dossier.
    """
    retrieval_engine.clear()
    semantic_cache.clear()
    audit_logger.clear()

    seed = client.post("/ingest-sample?strategy=parent_child")
    assert seed.status_code == 200

    docs_resp = client.get("/documents")
    assert docs_resp.status_code == 200
    docs_payload = docs_resp.json()
    assert docs_payload["count"] == 3
    first_doc = docs_payload["documents"][0]
    for key in ("doc_id", "title", "source", "tenant_id", "access_roles", "chunk_count"):
        assert key in first_doc

    doc_id_to_delete = first_doc["doc_id"]
    del_resp = client.delete(f"/documents/{doc_id_to_delete}")
    assert del_resp.status_code == 200

    docs_after = client.get("/documents").json()
    assert docs_after["count"] == 2
    assert all(d["doc_id"] != doc_id_to_delete for d in docs_after["documents"])

    audit_report_resp = client.get("/compliance/audit-report")
    assert audit_report_resp.status_code == 200
    report = audit_report_resp.json()
    assert report["chain_valid"] is True
    assert "gdpr_compliance_summary" in report
    assert "metrics_snapshot" in report
    assert "document_inventory" in report
    assert "markdown_dossier" in report
    assert "VALID" in report["markdown_dossier"]
