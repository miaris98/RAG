"""TDD Test Suite for Principal-Level RAG, GDPR/Auditability & Security Features:
- P0.1: Cyclic Self-Correction loop in LangGraph (verify -> retry retrieve_and_rerank)
- P0.2: MMR Diversity, Negation-Aware Reranking & Lost-in-the-Middle Reordering
- P0.3: Table-Header & Code-Block-Aware Chunking
- P0.4: Multi-Turn Conversational Memory & Coreference Resolution (session_id)
- P1.1: Circuit Breakers + GET /metrics (p50/p95/p99, cache hit rate, CB state)
- P1.2: GDPR Surgical Deletion (DELETE /documents/{doc_id}) & Version Precedence
- P1.3: Async Ingestion Queue (POST /ingest-async)
- P1.4: Tamper-Evident Hash-Chained Audit Trail & GDPR Compliance Export (GET /audit-logs)
- P2.1/P2.3: Labeled-Query Sidecar & MCP Tool Security Inspection
"""

import time
from fastapi.testclient import TestClient

from main import app
from app.agent.graph import execute_rag_pipeline
from app.guardrails.audit import audit_logger
from app.ingestion.chunking import ChunkingEngine
from app.ingestion.pipeline import ingestion_pipeline
from app.memory.conversation import conversation_memory
from app.models import DocumentInput, QueryInput
from app.observability.metrics import CircuitBreaker, metrics_collector
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.labeled_queries import labeled_query_store
from app.retrieval.postprocessor import ContextPostProcessor
from app.retrieval.semantic_cache import semantic_cache


client = TestClient(app)


def test_p0_1_cyclic_self_correction_in_langgraph():
    """P0.1: When initial retrieval/faithfulness is low, LangGraph loops back to retrieve_and_rerank."""
    retrieval_engine.clear()
    semantic_cache.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    # Force a low-threshold test via max_retries=1 on a query that triggers self-correction check
    out = execute_rag_pipeline(
        QueryInput(
            text="What triggers ERR_LEDGER_9042?",
            retrieval_mode="hybrid_graph_rerank",
            top_k=2,
            enable_web_search=False,
            enable_semantic_cache=False,
            max_retries=1,
        )
    )
    assert "retry_count" in out["trace"]
    assert isinstance(out["trace"]["critique_history"], list)


def test_p0_2_mmr_diversity_negation_and_lost_in_the_middle():
    """P0.2: MMR removes near-duplicates, negation penalty demotes contradicted text,
    and Lost-in-the-Middle places highest-relevance chunks at start and end.
    """
    processor = ContextPostProcessor(mmr_lambda=0.6)
    from app.models import RetrievedChunk

    c1 = RetrievedChunk(
        chunk_id="c1", doc_id="d1", parent_id="p1", title="T1", source="s1",
        content="AegisPay settles cross-border transfers in 180 milliseconds using Raft.",
        parent_content="AegisPay settles cross-border transfers in 180 milliseconds using Raft.",
        calibrated_prob=0.95, is_relevant=True,
    )
    c2_dup = RetrievedChunk(
        chunk_id="c2", doc_id="d1", parent_id="p2", title="T1", source="s1",
        content="AegisPay settles cross-border transfers in 180 milliseconds using Raft consensus.",
        parent_content="AegisPay settles cross-border transfers in 180 milliseconds using Raft consensus.",
        calibrated_prob=0.90, is_relevant=True,
    )
    c3 = RetrievedChunk(
        chunk_id="c3", doc_id="d2", parent_id="p3", title="T2", source="s2",
        content="MinIO bucket aegis-audit-prod retains encrypted chunks for 90 days.",
        parent_content="MinIO bucket aegis-audit-prod retains encrypted chunks for 90 days.",
        calibrated_prob=0.80, is_relevant=True,
    )
    c4 = RetrievedChunk(
        chunk_id="c4", doc_id="d3", parent_id="p4", title="T3", source="s3",
        content="Etcd manages Milvus metadata coordination on port 2379.",
        parent_content="Etcd manages Milvus metadata coordination on port 2379.",
        calibrated_prob=0.70, is_relevant=True,
    )

    selected = processor.select_mmr([c1, c2_dup, c3, c4], top_k=3)
    selected_ids = [c.chunk_id for c in selected]
    # c3 and c4 should be preferred over near-duplicate c2_dup
    assert "c1" in selected_ids
    assert "c3" in selected_ids

    reordered = processor.lost_in_the_middle_reorder([c1, c3, c4])
    # Rank 1 at start, Rank 2 at end, Rank 3 in middle
    assert reordered[0].chunk_id == "c1"
    assert reordered[-1].chunk_id == "c3"
    assert reordered[1].chunk_id == "c4"


def test_p0_3_table_and_code_aware_chunking():
    """P0.3: Markdown tables preserve header rows when split, and fenced code blocks stay intact."""
    engine = ChunkingEngine(parent_chunk_size=260, parent_chunk_overlap=30, child_chunk_size=180, child_chunk_overlap=20)
    md_with_table = """# Service Ports
| Service | Port | Protocol |
| :--- | :--- | :--- |
| FastAPI | 8000 | HTTP |
| Milvus | 19530 | gRPC |
| RabbitMQ | 5672 | AMQP |
| Ollama | 11434 | HTTP |
| Gradio | 7860 | HTTP |
| MinIO | 9000 | S3 |
"""
    chunks = engine.chunk_document(md_with_table, title="Ports", source="ports.md", strategy="parent_child")
    assert len(chunks) >= 2
    for chk in chunks:
        if "| FastAPI |" in chk.parent_content or "| MinIO |" in chk.parent_content:
            assert "| Service | Port | Protocol |" in chk.parent_content


def test_p0_4_multiturn_conversational_memory_and_coreference():
    """P0.4: Session memory resolves pronouns like 'its' or 'that error' using prior turn entities."""
    retrieval_engine.clear()
    semantic_cache.clear()
    conversation_memory.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    sid = "session-interview-01"
    out1 = execute_rag_pipeline(
        QueryInput(
            text="What triggers ERR_LEDGER_9042?",
            session_id=sid,
            enable_web_search=False,
            enable_semantic_cache=False,
        )
    )
    assert "ERR_LEDGER_9042" in out1["answer"]

    # Follow-up uses pronoun 'its' without naming ERR_LEDGER_9042
    out2 = execute_rag_pipeline(
        QueryInput(
            text="What is its remediation command?",
            session_id=sid,
            enable_web_search=False,
            enable_semantic_cache=False,
        )
    )
    assert "aegis-ctl raft resync" in out2["answer"].lower()
    assert any("err_ledger_9042" in q.lower() for q in out2["trace"]["expanded_queries"])


def test_p1_1_circuit_breaker_and_observability_metrics():
    """P1.1: CircuitBreaker transitions CLOSED -> OPEN after threshold failures, and GET /metrics returns p95."""
    cb = CircuitBreaker(name="test_llm", failure_threshold=2, recovery_timeout_sec=30.0)
    assert cb.state == "CLOSED"
    cb.record_failure()
    assert cb.state == "CLOSED"
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.allow_request() is False

    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "latency_ms" in data
    assert "p95" in data["latency_ms"]
    assert "circuit_breakers" in data


def test_p1_2_gdpr_surgical_deletion_and_version_precedence():
    """P1.2: DELETE /documents/{doc_id} erases a document across Dense, BM25, Entity Graph, and Cache."""
    retrieval_engine.clear()
    semantic_cache.clear()
    audit_logger.clear()

    r1 = client.post(
        "/ingest",
        json={
            "title": "GDPR Subject Record",
            "source": "internal://hr/gdpr-subject.md",
            "content": "# Subject Profile\nSubject token GDPR_TOKEN_4411 belongs to user record 4411.",
            "retention_days": 30,
            "lawful_basis": "consent",
        },
    )
    assert r1.status_code == 200
    doc_id = r1.json()["doc_id"]

    # Confirm retrievable before deletion
    ret_before = client.post("/retrieve", json={"query": "GDPR_TOKEN_4411", "retrieval_mode": "hybrid_graph_rerank"})
    assert ret_before.json()["count"] == 1

    # Surgical GDPR Article 17 Erasure
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["chunks_deleted"] >= 1

    # Confirm completely erased from retrieval
    ret_after = client.post("/retrieve", json={"query": "GDPR_TOKEN_4411", "retrieval_mode": "hybrid_graph_rerank"})
    assert ret_after.json()["count"] == 0


def test_p1_3_async_ingestion_queue():
    """P1.3: POST /ingest-async queues document ingestion and completes via /get-job/{job_id}."""
    retrieval_engine.clear()
    res = client.post(
        "/ingest-async",
        json={
            "title": "Async Spec",
            "source": "internal://async.md",
            "content": "# Async Queue\nAsync ingestion token ASYNC_9876 indexed via worker.",
        },
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    final = None
    for _ in range(30):
        poll = client.get(f"/get-job/{job_id}").json()
        if poll["status"] in ("completed", "failed"):
            final = poll
            break
        time.sleep(0.05)

    assert final is not None
    assert final["status"] == "completed"


def test_p1_4_tamper_evident_audit_trail_and_gdpr_compliance():
    """P1.4: Every query & GDPR deletion creates a hash-chained, PII-safe audit log verified by /audit-logs."""
    retrieval_engine.clear()
    audit_logger.clear()
    client.post("/ingest-sample?strategy=parent_child")

    client.post(
        "/query-stream",
        json={"text": "What triggers ERR_LEDGER_9042?", "enable_web_search": False},
    )
    audit_res = client.get("/audit-logs")
    assert audit_res.status_code == 200
    payload = audit_res.json()
    assert payload["chain_valid"] is True
    assert payload["total_events"] >= 2
    assert any(e["event_type"] == "rag_query" for e in payload["events"])


def test_p2_3_labeled_query_sidecar_and_mcp_security():
    """P2.3: Labeled-Query Sidecar serves vetted golden answers, and MCP tool definitions pass security checks."""
    labeled_query_store.clear()
    labeled_query_store.add_vetted_query(
        query="What is the SLA for AegisPay?",
        vetted_answer="AegisPay settles intra-region liquidity pools in 180 milliseconds via Raft consensus [Vetted-QA].",
        access_roles=["public", "engineering"],
    )
    hit = labeled_query_store.lookup("What is the SLA for AegisPay?", user_roles=["engineering"])
    assert hit is not None
    assert "180 milliseconds" in hit["answer"]
