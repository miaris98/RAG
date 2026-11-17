"""Pydantic data contracts for ingestion, hybrid retrieval, agent traces, and evaluation."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ChunkingStrategy = Literal["parent_child", "contextual_recursive", "fixed"]
RetrievalMode = Literal["dense", "sparse", "hybrid_rrf", "hybrid_rerank"]


class DocumentInput(BaseModel):
    """Input payload for document ingestion (backward-compatible with legacy {'content': '...'})."""

    content: str = Field(..., description="Raw document text to chunk and index.")
    title: str = Field(default="Untitled Document", description="Human-readable document title.")
    source: str = Field(default="manual_input", description="Origin URI, filename, or system identifier.")
    chunking_strategy: ChunkingStrategy = Field(
        default="parent_child",
        description="Chunking strategy: 'parent_child' (Small-to-Big), 'contextual_recursive', or 'fixed'.",
    )
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom metadata tags.")


class ChunkRecord(BaseModel):
    """Canonical representation of an indexed chunk across Milvus and BM25."""

    chunk_id: str
    doc_id: str
    parent_id: str
    title: str
    source: str
    section: str = "General"
    content: str  # Child chunk text (with contextual header) used for retrieval
    parent_content: str  # Expanded parent context window passed to the LLM
    chunk_index: int = 0
    token_estimate: int = 0
    chunking_strategy: str = "parent_child"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A candidate chunk scored by Dense, Sparse (BM25), RRF, and Cross-Encoder stages."""

    chunk_id: str
    doc_id: str
    parent_id: str
    title: str
    source: str
    section: str = "General"
    content: str
    parent_content: str
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    rrf_score: float = 0.0
    rerank_logit: Optional[float] = None
    calibrated_prob: float = 0.0
    is_relevant: bool = True


class QueryInput(BaseModel):
    """Input payload for RAG query jobs (backward-compatible with legacy {'text': '...'})."""

    text: str = Field(..., description="User question or information request.")
    retrieval_mode: RetrievalMode = Field(
        default="hybrid_rerank",
        description="Retrieval pipeline mode: 'dense', 'sparse', 'hybrid_rrf', or 'hybrid_rerank'.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of final chunks to pass to generation.")
    enable_query_expansion: bool = Field(
        default=True,
        description="Generate lexical + semantic sub-queries before hybrid retrieval.",
    )
    enable_web_search: bool = Field(
        default=True,
        description="Allow external DuckDuckGo search for comparison or CRAG fallback.",
    )
    enable_self_rag: bool = Field(
        default=True,
        description="Run post-generation faithfulness & citation verification.",
    )


class RetrieveDebugRequest(BaseModel):
    """Direct retrieval debugging request without LLM generation."""

    query: str
    retrieval_mode: RetrievalMode = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=20)


class PipelineTrace(BaseModel):
    """End-to-end observability telemetry attached to every completed RAG job."""

    original_query: str
    expanded_queries: List[str] = Field(default_factory=list)
    retrieval_mode: str = "hybrid_rerank"
    candidates_retrieved_count: int = 0
    relevant_chunks_count: int = 0
    crag_decision: str = "sufficient"
    self_rag_verdict: str = "grounded"
    faithfulness_score: float = 1.0
    citation_count: int = 0
    stage_latencies_ms: Dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    web_search_triggered: bool = False
    web_snippets_preview: str = ""


class EvaluationRequest(BaseModel):
    """Configuration for running the RAG retrieval & grounding ablation suite."""

    top_k: int = Field(default=3, ge=1, le=10)
    modes: List[RetrievalMode] = Field(
        default_factory=lambda: ["dense", "sparse", "hybrid_rrf", "hybrid_rerank"]
    )
    seed_golden_corpus: bool = Field(
        default=True,
        description="Ensure the golden technical benchmark documents are indexed before evaluating.",
    )
