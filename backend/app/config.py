"""Centralized configuration for the Staff RAG Engine.

Fixes the legacy bug where OLLAMA_BASE_URL was defined from the environment
but ignored in favor of a hardcoded 'http://host.docker.internal:11434/v1'.
"""

import os
from dataclasses import dataclass, field


@dataclass
class RAGSettings:
    """Environment-backed configuration for all RAG pipeline stages."""

    # Infrastructure endpoints
    rabbitmq_host: str = field(default_factory=lambda: os.getenv("RABBITMQ_HOST", "127.0.0.1"))
    rabbitmq_port: int = field(default_factory=lambda: int(os.getenv("RABBITMQ_PORT", "5672")))
    rabbitmq_queue: str = field(default_factory=lambda: os.getenv("RABBITMQ_QUEUE", "rag_jobs"))
    rabbitmq_dlq: str = field(default_factory=lambda: os.getenv("RABBITMQ_DLQ", "rag_jobs_dlq"))
    async_fallback_on_broker_down: bool = field(
        default_factory=lambda: os.getenv("ASYNC_FALLBACK_ON_BROKER_DOWN", "true").lower() == "true"
    )

    milvus_host: str = field(default_factory=lambda: os.getenv("MILVUS_HOST", "127.0.0.1"))
    milvus_port: str = field(default_factory=lambda: os.getenv("MILVUS_PORT", "19530"))
    milvus_collection: str = field(default_factory=lambda: os.getenv("MILVUS_COLLECTION", "rag_documents"))

    # LLM & Model Configuration (100% local Ollama by default, OpenAI-compatible override)
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0")))

    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )

    # Chunking Hyperparameters
    parent_chunk_size: int = field(default_factory=lambda: int(os.getenv("PARENT_CHUNK_SIZE", "1200")))
    parent_chunk_overlap: int = field(default_factory=lambda: int(os.getenv("PARENT_CHUNK_OVERLAP", "200")))
    child_chunk_size: int = field(default_factory=lambda: int(os.getenv("CHILD_CHUNK_SIZE", "380")))
    child_chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHILD_CHUNK_OVERLAP", "60")))

    # Retrieval & Reranking Hyperparameters
    retrieval_top_k_dense: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K_DENSE", "10")))
    retrieval_top_k_sparse: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K_SPARSE", "10")))
    rerank_top_k: int = field(default_factory=lambda: int(os.getenv("RERANK_TOP_K", "5")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    dense_weight: float = field(default_factory=lambda: float(os.getenv("DENSE_WEIGHT", "1.0")))
    sparse_weight: float = field(default_factory=lambda: float(os.getenv("SPARSE_WEIGHT", "1.0")))
    relevance_threshold: float = field(default_factory=lambda: float(os.getenv("RELEVANCE_THRESHOLD", "0.25")))

    @property
    def llm_api_base(self) -> str:
        """Resolves the OpenAI-compatible base URL from LLM_API_BASE or OLLAMA_BASE_URL."""
        explicit_base = os.getenv("LLM_API_BASE")
        if explicit_base:
            return explicit_base.rstrip("/")
        return f"{self.ollama_base_url.rstrip('/')}/v1"


settings = RAGSettings()
