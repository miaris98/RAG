"""Pydantic data contracts for ingestion, RBAC, hybrid/graph retrieval, agent traces, and evaluation."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ChunkingStrategy = Literal["parent_child", "contextual_recursive", "fixed"]
RetrievalMode = Literal["dense", "sparse", "hybrid_rrf", "hybrid_rerank", "hybrid_graph_rerank"]


class DocumentInput(BaseModel):
    """Input payload for document ingestion (backward-compatible with legacy {'content': '...'})."""

    content: str = Field(..., description="Raw document text to chunk and index.")
    title: str = Field(default="Untitled Document", description="Human-readable document title.")
    source: str = Field(default="manual_input", description="Origin URI, filename, or system identifier.")
    chunking_strategy: ChunkingStrategy = Field(
        default="parent_child",
        description="Chunking strategy: 'parent_child' (Small-to-Big), 'contextual_recursive', or 'fixed'.",
    )
    tenant_id: str = Field(default="default", description="Multi-tenant isolation namespace.")
    access_roles: List[str] = Field(
        default_factory=lambda: ["public"],
        description="Roles authorized to retrieve chunks from this document.",
    )
    apply_guardrails: bool = Field(
        default=True,
        description="Run PII redaction and indirect prompt injection sanitization before indexing.",
    )
    retention_days: int = Field(default=90, ge=1, le=3650, description="GDPR data retention TTL in days.")
    lawful_basis: str = Field(default="legitimate_interest", description="GDPR Article 6 lawful basis.")
    version: int = Field(default=1, ge=1, description="Document version for precedence ranking.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom metadata tags.")


class ChunkRecord(BaseModel):
    """Canonical representation of an indexed chunk across Milvus, BM25, and Entity Graph."""

    chunk_id: str
    doc_id: str
    parent_id: str
    title: str
    source: str
    section: str = "General"
    tenant_id: str = "default"
    access_roles: List[str] = Field(default_factory=lambda: ["public"])
    content: str  # Child chunk text (with contextual header) used for retrieval
    parent_content: str  # Expanded parent context window passed to the LLM
    chunk_index: int = 0
    token_estimate: int = 0
    chunking_strategy: str = "parent_child"
    entities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A candidate chunk scored by Dense, Sparse (BM25), Entity Graph, RRF, and Cross-Encoder stages."""

    chunk_id: str
    doc_id: str
    parent_id: str
    title: str
    source: str
    section: str = "General"
    tenant_id: str = "default"
    access_roles: List[str] = Field(default_factory=lambda: ["public"])
    content: str
    parent_content: str
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    graph_rank: Optional[int] = None
    graph_score: Optional[float] = None
    rrf_score: float = 0.0
    rerank_logit: Optional[float] = None
    calibrated_prob: float = 0.0
    is_relevant: bool = True
    matched_terms: List[str] = Field(default_factory=list)
    matched_entities: List[str] = Field(default_factory=list)
    rrf_breakdown: Dict[str, float] = Field(default_factory=dict)
    prompt_position: Optional[int] = None


class CitationItem(BaseModel):
    """Structured inline citation linking generated claims back to exact retrieved evidence."""

    ref_id: str
    doc_id: str
    chunk_id: str
    title: str
    source: str
    section: str = "General"
    calibrated_prob: float = 0.0
    supporting_quote: str = ""
    matched_terms: List[str] = Field(default_factory=list)
    matched_entities: List[str] = Field(default_factory=list)


class QueryInput(BaseModel):
    """Input payload for RAG query jobs (backward-compatible with legacy {'text': '...'})."""

    text: str = Field(..., description="User question or information request.")
    retrieval_mode: RetrievalMode = Field(
        default="hybrid_graph_rerank",
        description="Retrieval pipeline mode: 'dense', 'sparse', 'hybrid_rrf', 'hybrid_rerank', or 'hybrid_graph_rerank'.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of final chunks to pass to generation.")
    tenant_id: str = Field(default="default", description="Multi-tenant isolation namespace.")
    user_roles: List[str] = Field(
        default_factory=lambda: ["public", "engineering", "compliance"],
        description="Caller RBAC roles used for pre-filtered retrieval.",
    )
    enable_query_expansion: bool = Field(
        default=True,
        description="Use Adaptive Query Router (Lexical / HyDE / Multi-Hop Decomposition).",
    )
    enable_web_search: bool = Field(
        default=True,
        description="Allow external DuckDuckGo search for comparison or CRAG fallback.",
    )
    enable_self_rag: bool = Field(
        default=True,
        description="Run post-generation faithfulness & citation verification.",
    )
    enable_semantic_cache: bool = Field(
        default=True,
        description="Check role-scoped Semantic Query Cache before executing pipeline.",
    )
    enable_guardrails: bool = Field(
        default=True,
        description="Run PII redaction & prompt injection defense on incoming query.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for multi-turn conversational memory & coreference resolution.",
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Maximum cyclic Self-RAG retry loops if faithfulness falls below threshold.",
    )


class RetrieveDebugRequest(BaseModel):
    """Direct retrieval debugging request without LLM generation."""

    query: str
    retrieval_mode: RetrievalMode = "hybrid_graph_rerank"
    top_k: int = Field(default=5, ge=1, le=20)
    tenant_id: str = "default"
    user_roles: List[str] = Field(default_factory=lambda: ["public", "engineering", "compliance"])


class PipelineTrace(BaseModel):
    """End-to-end observability telemetry attached to every completed RAG job."""

    trace_id: Optional[str] = None
    original_query: str
    query_route: str = "standard"
    expanded_queries: List[str] = Field(default_factory=list)
    retrieval_mode: str = "hybrid_graph_rerank"
    tenant_id: str = "default"
    user_roles: List[str] = Field(default_factory=list)
    cache_hit: bool = False
    guardrail_flags: List[str] = Field(default_factory=list)
    candidates_retrieved_count: int = 0
    relevant_chunks_count: int = 0
    crag_decision: str = "sufficient"
    self_rag_verdict: str = "grounded"
    faithfulness_score: float = 1.0
    citation_count: int = 0
    retry_count: int = 0
    critique_history: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    stage_latencies_ms: Dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    retrieval_funnel: Dict[str, Any] = Field(default_factory=dict)
    citations: List[CitationItem] = Field(default_factory=list)
    waterfall_markdown: str = ""
    web_search_triggered: bool = False
    web_snippets_preview: str = ""


class EvaluationRequest(BaseModel):
    """Configuration for running the RAG retrieval & grounding ablation suite."""

    top_k: int = Field(default=3, ge=1, le=10)
    modes: List[RetrievalMode] = Field(
        default_factory=lambda: ["dense", "sparse", "hybrid_rrf", "hybrid_rerank", "hybrid_graph_rerank"]
    )
    seed_golden_corpus: bool = Field(
        default=True,
        description="Ensure the golden technical benchmark documents are indexed before evaluating.",
    )
