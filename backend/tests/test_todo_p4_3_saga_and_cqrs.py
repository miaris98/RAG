"""TDD tests for P4.3: Database-per-Service boundaries, Saga compensating transactions, and CQRS Read Model."""

from fastapi.testclient import TestClient

from main import app
from app.data.cqrs import cqrs_read_model
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_ingestion_saga_commits_and_rolls_back_via_compensating_transactions_on_failure():
    """Saga Pattern: successful saga indexes & updates CQRS; mid-saga failure runs compensating rollbacks leaving zero orphan chunks."""
    retrieval_engine.clear()
    cqrs_read_model.clear()

    # 1. Happy path Saga commit
    r_ok = client.post(
        "/saga/ingest",
        json={
            "title": "Saga Commit Spec",
            "content": "Committed document under transactional saga coordination.",
            "tenant_id": "tenant_saga",
            "access_roles": ["public", "engineering"],
        },
    )
    assert r_ok.status_code == 200
    d_ok = r_ok.json()
    assert d_ok["saga_status"] == "committed"
    assert d_ok["compensated_steps"] == []
    assert d_ok["doc_id"]

    # 2. Simulated failure at audit step -> compensates vector index + CQRS read projection!
    r_fail = client.post(
        "/saga/ingest?inject_failure_step=record_audit_lineage",
        json={
            "title": "Saga Rollback Spec",
            "content": "Orphan content that MUST be rolled back when audit step fails.",
            "tenant_id": "tenant_saga",
            "access_roles": ["public"],
        },
    )
    assert r_fail.status_code == 200
    d_fail = r_fail.json()
    assert d_fail["saga_status"] == "compensated"
    assert "index_vector_and_lexical" in d_fail["compensated_steps"]
    assert "update_cqrs_projection" in d_fail["compensated_steps"]

    # Verify zero orphan chunks leaked into retrieval index or CQRS read model
    ret = client.post(
        "/retrieve",
        json={"query": "Orphan content rolled back", "tenant_id": "tenant_saga", "user_roles": ["public"]},
    )
    assert ret.status_code == 200
    assert not any("Saga Rollback Spec" in c["title"] for c in ret.json()["results"])


def test_cqrs_read_model_projection_and_database_per_service_boundaries():
    """CQRS serves lock-free materialized read views updated by write commands; /data/boundaries enforces DB-per-service."""
    retrieval_engine.clear()
    cqrs_read_model.clear()

    client.post(
        "/ingest",
        json={
            "title": "CQRS Projected Policy",
            "content": "CQRS separates write-path commands from materialized read projections.",
            "tenant_id": "tenant_cqrs",
            "access_roles": ["public", "compliance"],
        },
    )

    rm = client.get("/cqrs/read-model")
    assert rm.status_code == 200
    data = rm.json()
    assert data["total_documents"] >= 1
    assert data["projection_version"] >= 1
    assert data["documents_by_tenant"].get("tenant_cqrs", 0) >= 1
    assert data["documents_by_role"].get("compliance", 0) >= 1

    boundaries = client.get("/data/boundaries")
    assert boundaries.status_code == 200
    b_data = boundaries.json()
    owners = {item["service"] for item in b_data["boundaries"]}
    assert {"retrieval_service", "sql_opsbot_service", "compliance_audit_service", "feedback_service"}.issubset(owners)
