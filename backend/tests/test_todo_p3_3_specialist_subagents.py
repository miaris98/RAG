"""TDD tests for P3.3: Parallel Specialist Subagents, least-privilege scoping, and Supervisor Orchestration."""

from fastapi.testclient import TestClient

from main import app
from app.ingestion.pipeline import ingestion_pipeline
from app.observability.tracing import trace_collector
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_list_registered_specialist_subagents_and_least_privilege_scopes():
    """GET /agent/subagents advertises all 4 least-privilege specialist subagents and their allowed tools."""
    resp = client.get("/agent/subagents")
    assert resp.status_code == 200
    data = resp.json()
    names = {s["agent_id"] for s in data["subagents"]}
    assert names == {
        "retrieval_specialist",
        "sql_ops_specialist",
        "compliance_auditor",
        "web_researcher",
    }
    by_id = {s["agent_id"]: s for s in data["subagents"]}
    # Verify least-privilege separation: sql_ops_specialist has no vector/compliance doc access
    assert "execute_readonly_sql" in by_id["sql_ops_specialist"]["allowed_tools"]
    assert "retrieve_hybrid" not in by_id["sql_ops_specialist"]["allowed_tools"]
    assert "retrieve_hybrid" in by_id["retrieval_specialist"]["allowed_tools"]


def test_supervisor_fans_out_parallel_subagents_and_records_waterfall_spans():
    """POST /agent/orchestrate runs retrieval + sql_ops + compliance_auditor, produces citations, and logs TraceCollector spans."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    resp = client.post(
        "/agent/orchestrate",
        json={
            "query": "Check live service inventory status and active incidents for AegisPay ERR_LEDGER_9042 and give the remediation command.",
            "tenant_id": "default",
            "user_roles": ["public", "engineering"],
            "enable_web_search": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    dispatched = [s["agent_id"] for s in data["subagent_results"]]
    assert "retrieval_specialist" in dispatched
    assert "sql_ops_specialist" in dispatched
    assert "compliance_auditor" in dispatched

    assert "ERR_LEDGER_9042" in data["final_answer"]
    assert "aegispay-ledger" in data["final_answer"].lower() or "inc-9042" in data["final_answer"].lower()
    assert len(data["citations"]) >= 1
    assert data["compliance_audit"]["verdict"] in ("pass", "fail")
    assert data["trace_id"]
    assert data["waterfall_markdown"]

    # Verify per-subagent spans were recorded in TraceCollector
    stored_trace = trace_collector.get_trace(data["trace_id"])
    assert stored_trace is not None
    span_names = [sp["name"] for sp in stored_trace["spans"]]
    assert "supervisor_orchestrator" in span_names
    assert "subagent.retrieval_specialist" in span_names
    assert "subagent.sql_ops_specialist" in span_names
    assert "subagent.compliance_auditor" in span_names


def test_supervisor_enforces_rbac_isolation_across_subagents():
    """Engineering-only caller must not leak compliance-restricted Governance data via any subagent."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    resp = client.post(
        "/agent/orchestrate",
        json={
            "query": "What is the EMEA sovereign compliance PagerDuty rotation ID?",
            "tenant_id": "default",
            "user_roles": ["engineering"],
            "enable_web_search": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "PD-EMEA-8821" not in data["final_answer"]
    assert not any("governance" in c["title"].lower() for c in data["citations"])
