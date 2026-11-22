"""TDD Test Suite for End-to-End Hierarchical Distributed Tracing, Deep Retrieval Visibility,
and Structured Inline Citations with Evidence Quote Extraction.
"""

from fastapi.testclient import TestClient

from main import app
from app.agent.graph import execute_rag_pipeline
from app.ingestion.pipeline import ingestion_pipeline
from app.models import QueryInput
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.semantic_cache import semantic_cache


client = TestClient(app)


def test_end_to_end_hierarchical_distributed_tracing_and_waterfall():
    """Step 1.1: Distributed trace records root span, stage spans, and nested retrieval sub-spans
    plus a visual waterfall timeline.
    """
    retrieval_engine.clear()
    semantic_cache.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    out = execute_rag_pipeline(
        QueryInput(
            text="What triggers ERR_LEDGER_9042 and what is the remediation command?",
            retrieval_mode="hybrid_graph_rerank",
            top_k=3,
            enable_web_search=False,
            enable_semantic_cache=False,
        )
    )
    trace_id = out["trace"].get("trace_id")
    assert trace_id is not None

    waterfall = out["trace"].get("waterfall_markdown", "")
    assert "█" in waterfall

    resp = client.get(f"/traces/{trace_id}")
    assert resp.status_code == 200
    t_rec = resp.json()
    spans = t_rec["spans"]

    root_spans = [s for s in spans if s.get("parent_span_id") is None and s["name"] == "rag_query_pipeline"]
    assert len(root_spans) == 1
    root_span_id = root_spans[0]["span_id"]

    child_stage_spans = [s for s in spans if s.get("parent_span_id") == root_span_id]
    assert len(child_stage_spans) >= 4

    span_names = {s["name"] for s in spans}
    for sub_name in (
        "retrieval.rbac_prefilter",
        "retrieval.dense_vector_search",
        "retrieval.sparse_bm25_search",
        "retrieval.entity_graph_traversal",
        "retrieval.rrf_and_parent_promotion",
        "retrieval.cross_encoder_and_mmr",
    ):
        assert sub_name in span_names, f"Missing nested retrieval sub-span: {sub_name}"

    for s in spans:
        assert "start_offset_ms" in s
        assert "duration_ms" in s
        assert "attributes" in s


def test_deep_retrieval_visibility_funnel_and_chunk_explainability():
    """Step 1.2: Retrieval funnel exposes per-stage candidate counts, pruned chunks,
    matched terms/entities, RRF breakdown, and Lost-in-the-Middle prompt positions.
    """
    retrieval_engine.clear()
    semantic_cache.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    out = execute_rag_pipeline(
        QueryInput(
            text="What triggers ERR_LEDGER_9042 and what is the remediation command?",
            retrieval_mode="hybrid_graph_rerank",
            top_k=3,
            enable_web_search=False,
            enable_semantic_cache=False,
        )
    )
    funnel = out["trace"].get("retrieval_funnel") or {}
    for key in (
        "total_corpus_chunks",
        "rbac_authorized_chunks",
        "dense_candidates_count",
        "sparse_candidates_count",
        "graph_candidates_count",
        "rrf_fused_pool_count",
        "parent_promoted_count",
        "crag_relevant_count",
        "mmr_selected_count",
        "pruned_chunks",
        "substage_latencies_ms",
    ):
        assert key in funnel, f"Missing retrieval_funnel key: {key}"

    assert funnel["total_corpus_chunks"] >= 10
    assert funnel["crag_relevant_count"] >= 1

    chunks = out["trace"]["retrieved_chunks"]
    assert len(chunks) >= 1
    top_chunk = chunks[0]
    assert "err_ledger_9042" in top_chunk.get("matched_terms", [])
    assert len(top_chunk.get("matched_entities", [])) >= 1
    assert set(top_chunk.get("rrf_breakdown", {}).keys()) == {"dense", "sparse", "graph"}
    relevant_positions = [c.get("prompt_position") for c in chunks if c.get("is_relevant")]
    assert all(isinstance(pos, int) and pos >= 1 for pos in relevant_positions)


def test_structured_inline_citations_and_evidence_quotes():
    """Step 1.3: Response contains inline [Doc i] badges, verified evidence bibliography,
    and structured CitationItem objects with exact supporting quotes.
    """
    retrieval_engine.clear()
    semantic_cache.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    out = execute_rag_pipeline(
        QueryInput(
            text="What triggers ERR_LEDGER_9042 and what is the remediation command?",
            retrieval_mode="hybrid_graph_rerank",
            top_k=3,
            enable_web_search=False,
            enable_semantic_cache=False,
        )
    )
    citations = out["trace"].get("citations") or []
    assert len(citations) >= 1
    first_cit = citations[0]
    assert first_cit["ref_id"] == "[Doc 1]"
    for field in ("doc_id", "chunk_id", "title", "source", "section", "calibrated_prob", "supporting_quote", "matched_terms"):
        assert field in first_cit
    quote_lower = first_cit["supporting_quote"].lower()
    assert ("err_ledger_9042" in quote_lower) or ("aegis-ctl" in quote_lower)

    answer = out["answer"]
    assert "[Doc 1" in answer
    assert "### 📚 Verified Inline Citations & Evidence Quotes" in answer
