"""TDD tests for Branch 8: Agoda structured SQL live inventory / OpsBot tool (#25) alongside vector RAG."""

from fastapi.testclient import TestClient

from main import app
from app.agent.graph import execute_rag_pipeline
from app.agent.sql_tool import sql_inventory_tool
from app.ingestion.pipeline import ingestion_pipeline
from app.models import QueryInput
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_agoda_sql_tool_blocks_mutations_and_answers_live_inventory_queries():
    """#25 Agoda OpsBot: read-only SQL tool serves live service/incident inventory and blocks SQL write attempts."""
    status_resp = client.get("/inventory/sql-status")
    assert status_resp.status_code == 200
    s_data = status_resp.json()
    assert len(s_data["services"]) >= 4
    assert len(s_data["active_incidents"]) >= 2

    # Valid read-only SELECT works
    ok_resp = client.post(
        "/inventory/sql-query",
        json={"sql": "SELECT service_name, region, status, active_nodes FROM service_inventory ORDER BY service_name"},
    )
    assert ok_resp.status_code == 200
    assert ok_resp.json()["row_count"] >= 4

    # Destructive SQL injection must be blocked with 400
    for bad_sql in (
        "DROP TABLE service_inventory",
        "DELETE FROM active_incidents",
        "SELECT * FROM service_inventory; DROP TABLE active_incidents",
        "UPDATE service_inventory SET status='down'",
    ):
        bad_resp = client.post("/inventory/sql-query", json={"sql": bad_sql})
        assert bad_resp.status_code == 400


def test_langgraph_agent_combines_live_sql_inventory_with_vector_rag():
    """#25 Agoda: asking for live service inventory + active incidents invokes sql_live_inventory_tool inside RAG."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    out = execute_rag_pipeline(
        QueryInput(
            text="Show live service inventory status and active incidents for AegisPay ERR_LEDGER_9042",
            retrieval_mode="hybrid_graph_rerank",
            top_k=3,
            enable_web_search=False,
            enable_semantic_cache=False,
        )
    )
    assert "sql_live_inventory_tool" in out["tools"]
    assert "aegispay-ledger" in out["answer"].lower() or "inc-9042" in out["answer"].lower()
    assert "ERR_LEDGER_9042" in out["answer"]


def test_all_50_company_matrix_rows_have_zero_pending_todo_markers():
    """Verifies docs/INDUSTRY_RAG_TODO.md has every Section 1 & Section 2 TODO completed cleanly."""
    from pathlib import Path

    todo_path = Path(__file__).resolve().parents[2] / "docs" / "INDUSTRY_RAG_TODO.md"
    content = todo_path.read_text(encoding="utf-8")
    assert "🔲 TODO" not in content
    assert "- [ ]" not in content
