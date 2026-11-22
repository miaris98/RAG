"""Typed state schema for the LangGraph CRAG + Self-RAG workflow."""

from typing import Any, Dict, List, TypedDict


class RAGGraphState(TypedDict, total=False):
    """State dictionary flowing through the LangGraph StateGraph nodes."""

    # Input configuration & RBAC context
    query: str
    retrieval_mode: str
    top_k: int
    tenant_id: str
    user_roles: List[str]
    enable_query_expansion: bool
    enable_web_search: bool
    enable_self_rag: bool
    enable_guardrails: bool
    session_id: str
    max_retries: int

    # Cyclic Self-Correction State
    retry_count: int
    critique_history: List[str]

    # Security Guardrails
    guardrail_flags: List[str]

    # Stage 1: Adaptive Query Router (Lexical / HyDE / Multi-Hop)
    query_route: str  # "lexical_lookup" | "conceptual_hyde" | "multi_hop" | "standard"
    expanded_queries: List[str]
    dense_weight: float
    sparse_weight: float

    # Stage 2: Tri-Modal Hybrid Retrieval & Cross-Encoder Reranking
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    retrieval_funnel: Dict[str, Any]

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
    citations: List[Dict[str, Any]]

    # Telemetry & Tool Audit
    tools_used: List[str]
    stage_latencies_ms: Dict[str, float]
