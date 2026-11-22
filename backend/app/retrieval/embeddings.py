"""Thread-safe singleton registry for Bi-Encoder Embeddings, Cross-Encoder Reranker, and LLM client.

Fixes two major anti-patterns from the legacy prototype:
1. Re-initializing HuggingFaceEmbeddings("all-MiniLM-L6-v2") inside document_retriever on every query.
2. Hardcoding 'http://host.docker.internal:11434/v1' instead of honoring OLLAMA_BASE_URL.
"""

import hashlib
import math
import threading
from typing import List, Optional

from app.config import settings


class DeterministicFallbackEmbeddings:
    """Fast deterministic n-gram hashing embedder (384-dim) used only if HuggingFace weights cannot load."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = [t.lower().strip(".,!?;:()[]{}\"'") for t in text.split() if t.strip()]
        if not tokens:
            return vec
        # Unigrams + character trigrams for semantic/morphological resilience
        features: List[str] = list(tokens)
        for tok in tokens:
            if len(tok) >= 4:
                features.extend(tok[i : i + 3] for i in range(len(tok) - 2))
        for feat in features:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class ModelRegistry:
    """Process-wide singleton holder for heavy ML models and LLM clients."""

    _lock = threading.Lock()
    _embeddings = None
    _embeddings_backend: str = "uninitialized"
    _reranker = None
    _reranker_backend: str = "uninitialized"
    _llm = None

    @classmethod
    def get_embeddings(cls):
        """Returns the singleton Bi-Encoder embedding model (all-MiniLM-L6-v2)."""
        if cls._embeddings is None:
            with cls._lock:
                if cls._embeddings is None:
                    try:
                        from langchain_huggingface import HuggingFaceEmbeddings

                        cls._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
                        cls._embeddings_backend = f"HuggingFaceEmbeddings({settings.embedding_model})"
                    except Exception as exc:
                        print(f"[ModelRegistry] Falling back to DeterministicFallbackEmbeddings: {exc}")
                        cls._embeddings = DeterministicFallbackEmbeddings(dim=384)
                        cls._embeddings_backend = "DeterministicFallbackEmbeddings(384)"
        return cls._embeddings

    @classmethod
    def get_reranker(cls):
        """Returns the singleton Cross-Encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)."""
        if cls._reranker is None and cls._reranker_backend == "uninitialized":
            with cls._lock:
                if cls._reranker is None and cls._reranker_backend == "uninitialized":
                    try:
                        from sentence_transformers import CrossEncoder

                        cls._reranker = CrossEncoder(settings.reranker_model)
                        cls._reranker_backend = f"CrossEncoder({settings.reranker_model})"
                    except Exception as exc:
                        print(f"[ModelRegistry] CrossEncoder fallback activated: {exc}")
                        cls._reranker = None
                        cls._reranker_backend = "LexicalSemanticCrossScorerFallback"
        return cls._reranker

    @classmethod
    def get_llm(cls):
        """Returns the singleton ChatOpenAI client wired to settings.llm_api_base (Ollama / OpenAI)."""
        if cls._llm is None:
            with cls._lock:
                if cls._llm is None:
                    from langchain_openai.chat_models import ChatOpenAI

                    cls._llm = ChatOpenAI(
                        temperature=settings.llm_temperature,
                        openai_api_base=settings.llm_api_base,
                        openai_api_key=settings.llm_api_key,
                        model_name=settings.llm_model,
                        max_retries=1,
                        request_timeout=45,
                    )
        return cls._llm

    @classmethod
    def status(cls) -> dict:
        """Returns diagnostic metadata about currently loaded models."""
        return {
            "embedding_model": settings.embedding_model,
            "embeddings_backend": cls._embeddings_backend,
            "reranker_model": settings.reranker_model,
            "reranker_backend": cls._reranker_backend,
            "llm_model": settings.llm_model,
            "llm_api_base": settings.llm_api_base,
        }
