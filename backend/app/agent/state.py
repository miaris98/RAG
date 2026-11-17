"""Typed state schema for the LangGraph CRAG + Self-RAG workflow."""

from typing import Any, Dict, List, TypedDict


class RAGGraphState(TypedDict, total=False):
    """State dictionary flowing through the LangGraph StateGraph nodes."""

    # Input configuration
    query: str
    retrieval_mode: str
    top_k: int
    enable_query_expansion: bool
    enable_web_search: bool
    enable_self_rag: bool

    # Stage 1: Query Analysis & Multi-Query Expansion
    expanded_queries: List[str]

    # Stage 2: Two-Stage Hybrid Retrieval & Cross-Encoder Reranking
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]

    # Stage 3: Corrective RAG (CRAG) Gate
    crag_decision: str  # "sufficient" | "ambiguous_web_augmented" | "insufficient_web_fallback"

    # Stage 4: External Web Search
    web_search_triggered: bool
    web_results: str

    # Stage 5: Grounded Cited Generation
    draft_answer: str
    final_answer: str

    # Stage 6: Self-RAG Faithfulness & Citation Verification
    self_rag_verdict: str  # "grounded" | "revised" | "no_internal_context" | "skipped"
    faithfulness_score: float
    citation_count: int

    # Telemetry & Tool Audit
    tools_used: List[str]
    stage_latencies_ms: Dict[str, float]
