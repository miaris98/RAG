"""TDD tests for Branch 3: Shopify/Notion CDC upsert, Reddit/Yelp freshness decay, and Zendesk user feedback loop."""

import time
from fastapi.testclient import TestClient

from main import app
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_cdc_webhook_incremental_upsert_and_delete():
    """Shopify/Notion CDC pattern: upsert replaces prior external_id chunks without leaving stale text; delete removes it."""
    retrieval_engine.clear()

    # 1. Initial CDC upsert (v1)
    r1 = client.post(
        "/ingest/cdc",
        json={
            "external_id": "catalog-sku-7741",
            "action": "upsert",
            "title": "Checkout Rate Limit Policy",
            "source": "cdc://shopify/policy-7741",
            "content": "Checkout API burst limit is 250 requests per second under v1 policy.",
            "tenant_id": "default",
            "access_roles": ["public", "engineering"],
            "version": 1,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["status"] == "success"
    assert d1["action"] == "upsert"
    assert d1["superseded_doc_ids"] == []
    first_doc_id = d1["doc_id"]

    # 2. CDC upsert update (v2) with same external_id -> supersedes v1 completely
    r2 = client.post(
        "/ingest/cdc",
        json={
            "external_id": "catalog-sku-7741",
            "action": "upsert",
            "title": "Checkout Rate Limit Policy",
            "source": "cdc://shopify/policy-7741",
            "content": "Checkout API burst limit is 900 requests per second under v2 policy.",
            "tenant_id": "default",
            "access_roles": ["public", "engineering"],
            "version": 2,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert first_doc_id in d2["superseded_doc_ids"]

    ret = client.post(
        "/retrieve",
        json={"query": "What is the Checkout API burst limit?", "retrieval_mode": "hybrid_rerank", "top_k": 5},
    )
    assert ret.status_code == 200
    combined = " ".join(c["parent_content"] for c in ret.json()["results"])
    assert "900 requests per second" in combined
    assert "250 requests per second" not in combined

    # 3. CDC delete -> removes v2 cleanly
    r3 = client.post(
        "/ingest/cdc",
        json={
            "external_id": "catalog-sku-7741",
            "action": "delete",
            "tenant_id": "default",
        },
    )
    assert r3.status_code == 200
    assert r3.json()["deleted_chunks"] >= 1

    ret_after = client.post(
        "/retrieve",
        json={"query": "What is the Checkout API burst limit?", "retrieval_mode": "hybrid_rerank", "top_k": 5},
    )
    assert ret_after.status_code == 200
    assert ret_after.json()["count"] == 0


def test_freshness_time_decay_boosts_recent_document_over_stale_equivalent():
    """Reddit/Yelp pattern: among equally matching documents, fresher updated_at outranks stale content."""
    retrieval_engine.clear()
    now = time.time()
    stale_epoch = now - (365 * 86400)  # 1 year old

    client.post(
        "/ingest",
        json={
            "title": "Latency Runbook (Old Archive)",
            "source": "internal://old-latency.md",
            "content": "Primary payment gateway timeout threshold is configured for canary cluster alpha.",
            "updated_at": stale_epoch,
            "version": 1,
            "access_roles": ["public"],
        },
    )
    client.post(
        "/ingest",
        json={
            "title": "Latency Runbook (Current 2026)",
            "source": "internal://fresh-latency.md",
            "content": "Primary payment gateway timeout threshold is configured for canary cluster beta.",
            "updated_at": now,
            "version": 1,
            "access_roles": ["public"],
        },
    )

    ret = client.post(
        "/retrieve",
        json={
            "query": "Primary payment gateway timeout threshold configured for canary cluster",
            "retrieval_mode": "hybrid_rrf",
            "top_k": 2,
        },
    )
    assert ret.status_code == 200
    results = ret.json()["results"]
    assert len(results) == 2
    assert results[0]["title"] == "Latency Runbook (Current 2026)"
    assert results[0]["freshness_boost"] > results[1]["freshness_boost"]


def test_user_feedback_loop_api_and_prometheus_metrics():
    """Zendesk pattern: POST /feedback records thumbs-up/down, GET /feedback summarizes, and Prometheus exports it."""
    r_up = client.post(
        "/feedback",
        json={
            "rating": "up",
            "query": "What triggers ERR_LEDGER_9042?",
            "trace_id": "tr_demo_001",
            "chunk_ids": ["chk_abc"],
            "user_roles": ["public", "engineering"],
            "comment": "Exact command cited cleanly.",
        },
    )
    assert r_up.status_code == 200
    assert r_up.json()["status"] == "recorded"

    r_down = client.post(
        "/feedback",
        json={
            "rating": "down",
            "query": "How do I rotate HSM keys?",
            "trace_id": "tr_demo_002",
            "comment": "Needed more detail on RabbitMQ prefetch timeout.",
        },
    )
    assert r_down.status_code == 200

    summary = client.get("/feedback")
    assert summary.status_code == 200
    data = summary.json()
    assert data["thumbs_up"] >= 1
    assert data["thumbs_down"] >= 1
    assert 0.0 <= data["satisfaction_rate"] <= 1.0
    assert len(data["recent_feedback"]) >= 2

    prom = client.get("/metrics/prometheus")
    assert prom.status_code == 200
    assert 'rag_user_feedback_total{rating="up"}' in prom.text
    assert 'rag_user_feedback_total{rating="down"}' in prom.text
    assert "rag_user_satisfaction_rate" in prom.text
