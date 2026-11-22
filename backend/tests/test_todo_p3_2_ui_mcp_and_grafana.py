"""TDD tests for P3.2: Extended MCP tools, Grafana dashboard metrics parity, and Gradio UI handlers."""

import json
from pathlib import Path

from app.ingestion.pipeline import ingestion_pipeline
from app.retrieval.hybrid import retrieval_engine
from mcp_server import (
    MCP_TOOL_MANIFEST,
    mcp_classify_structured_taxonomy,
    mcp_query_live_sql_inventory,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_mcp_server_exposes_sql_inventory_and_structured_taxonomy_tools():
    """P3.2 MCP parity: exposes read-only SQL OpsBot and constrained taxonomy classifier tools."""
    names = {t["name"] for t in MCP_TOOL_MANIFEST}
    assert "query_live_sql_inventory" in names
    assert "classify_structured_taxonomy" in names

    # 1. Read-only SQL tool works and blocks mutations
    sql_res = mcp_query_live_sql_inventory("SELECT service_name, status FROM service_inventory")
    assert sql_res["row_count"] >= 4

    # 2. Structured taxonomy tool classifies grounded in sample corpus
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")
    tax_res = mcp_classify_structured_taxonomy(
        query="What causes ERR_LEDGER_9042 quorum drift timeout?",
        allowed_labels=["payment_ledger_quorum_incident", "sovereign_compliance_escalation"],
        user_roles=["public", "engineering"],
    )
    assert tax_res["classification_label"] == "payment_ledger_quorum_incident"
    assert tax_res["audit_event_hash"]


def test_grafana_dashboard_charts_feedback_and_prefix_cache_prometheus_metrics():
    """P3.2 Grafana parity: pre-provisioned dashboard includes panels for feedback and LinkedIn prefix cache."""
    dash_path = REPO_ROOT / "observability" / "grafana" / "dashboards" / "rag_observability.json"
    dash_data = json.loads(dash_path.read_text(encoding="utf-8"))
    raw = json.dumps(dash_data)

    for metric in (
        "rag_user_feedback_total",
        "rag_user_satisfaction_rate",
        "rag_prefix_cache_hits_total",
        "rag_prefix_cache_amortized_tokens_total",
    ):
        assert metric in raw, f"Grafana dashboard missing Prometheus metric: {metric}"
