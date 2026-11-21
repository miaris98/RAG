"""TDD tests for Branch 7: LinkedIn Shared-Prefix Prompt Caching (#12) & Instacart Head-Query Pre-Warming (#22)."""

from fastapi.testclient import TestClient

from main import app
from app.agent.graph import execute_rag_pipeline
from app.ingestion.pipeline import ingestion_pipeline
from app.models import QueryInput
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.prefix_cache import prefix_cache
from app.retrieval.semantic_cache import semantic_cache


client = TestClient(app)


def test_linkedin_shared_prefix_prompt_cache_reuses_kv_prefix_and_exports_prometheus():
    """#12 LinkedIn: identical system+context prefix with different questions hits prefix cache and amortizes tokens."""
    prefix_cache.clear()

    context_blocks = [
        "[Doc 1] AegisPay RFC-402: ERR_LEDGER_9042 occurs when quorum lease expires. Run `aegis-ctl raft resync`."
    ]
    p1 = prefix_cache.build_prompt(
        context_blocks=context_blocks,
        user_query="What triggers ERR_LEDGER_9042?",
        web_results="None",
    )
    assert p1["prefix_cache_hit"] is False
    # Verify shared prefix (system + context) precedes dynamic user question for KV-cache reuse
    assert p1["full_prompt"].index("VERIFIED INTERNAL DOCUMENT CHUNKS") < p1["full_prompt"].index("USER QUESTION:")

    p2 = prefix_cache.build_prompt(
        context_blocks=context_blocks,
        user_query="What is the remediation command for ERR_LEDGER_9042?",
        web_results="None",
    )
    assert p2["prefix_cache_hit"] is True
    assert p2["prefix_hash"] == p1["prefix_hash"]
    assert p2["amortized_tokens"] > 0

    stats = prefix_cache.stats()
    assert stats["prefix_hits"] >= 1
    assert stats["amortized_prefix_tokens_total"] > 0

    prom = client.get("/metrics/prometheus")
    assert prom.status_code == 200
    assert "rag_prefix_cache_hits_total" in prom.text
    assert "rag_prefix_cache_amortized_tokens_total" in prom.text


def test_instacart_head_query_cache_prewarming_serves_subsequent_head_queries_from_cache():
    """#22 Instacart: POST /cache/prewarm warms semantic_cache so subsequent live queries get instant cache_hit=True."""
    retrieval_engine.clear()
    semantic_cache.clear()
    prefix_cache.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    head_queries = [
        "What triggers ERR_LEDGER_9042 and what is the remediation command?",
        "What is the p95 retrieval latency SLO for production RAG?",
    ]
    resp = client.post(
        "/cache/prewarm",
        json={
            "queries": head_queries,
            "tenant_id": "default",
            "user_roles": ["public", "engineering"],
            "retrieval_mode": "hybrid_graph_rerank",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["prewarmed_count"] == 2

    # Subsequent live execution of the head query must hit semantic_cache immediately
    live_out = execute_rag_pipeline(
        QueryInput(
            text=head_queries[0],
            tenant_id="default",
            user_roles=["public", "engineering"],
            retrieval_mode="hybrid_graph_rerank",
            enable_web_search=False,
            enable_semantic_cache=True,
        )
    )
    assert live_out["trace"]["cache_hit"] is True

    c_stats = client.get("/cache/stats")
    assert c_stats.status_code == 200
    assert "prefix_cache" in c_stats.json()
    assert "semantic_cache" in c_stats.json()
