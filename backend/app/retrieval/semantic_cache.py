"""Role-Scoped Semantic Query Cache with Ingestion-Aware Cache Invalidation.

Why this matters in Staff RAG Interviews:
- A naive query cache keys only on exact string match and ignores RBAC roles or document updates.
- Our SemanticQueryCache:
  1. Keys entries by (tenant_id, sorted user_roles, retrieval_mode, enable_web_search).
  2. Computes cosine similarity between the incoming query embedding and cached queries (>= 0.92).
  3. Tracks contributing `doc_id`s per cache entry so re-ingesting/updating a document
     automatically invalidates only affected cache entries.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.retrieval.embeddings import ModelRegistry


def _cosine(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    return (dot / (na * nb)) if (na > 0.0 and nb > 0.0) else 0.0


@dataclass
class SemanticCacheEntry:
    cache_key_scope: Tuple[str, Tuple[str, ...], str, bool]
    query_text: str
    query_vector: List[float]
    response_payload: Dict[str, Any]
    doc_ids: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)


class SemanticQueryCache:
    """Thread-safe semantic query cache with RBAC scoping and doc-level invalidation."""

    def __init__(self, similarity_threshold: float = 0.92, max_entries: int = 256):
        self._lock = threading.RLock()
        self._entries: List[SemanticCacheEntry] = []
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_scope(
        tenant_id: str,
        user_roles: List[str],
        retrieval_mode: str,
        enable_web_search: bool,
    ) -> Tuple[str, Tuple[str, ...], str, bool]:
        normalized_roles = tuple(sorted({r.strip().lower() for r in user_roles if r and r.strip()}))
        return (tenant_id, normalized_roles, retrieval_mode, bool(enable_web_search))

    def lookup(
        self,
        query: str,
        tenant_id: str,
        user_roles: List[str],
        retrieval_mode: str,
        enable_web_search: bool,
    ) -> Optional[Dict[str, Any]]:
        """Finds a semantically matching cached response within the exact same RBAC & mode scope."""
        scope = self._make_scope(tenant_id, user_roles, retrieval_mode, enable_web_search)
        embedder = ModelRegistry.get_embeddings()
        q_vec = embedder.embed_query(query)

        with self._lock:
            best_entry: Optional[SemanticCacheEntry] = None
            best_sim = -1.0

            for entry in self._entries:
                if entry.cache_key_scope != scope:
                    continue
                if entry.query_text.strip().lower() == query.strip().lower():
                    best_entry = entry
                    best_sim = 1.0
                    break
                sim = _cosine(q_vec, entry.query_vector)
                if sim >= self.similarity_threshold and sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry is not None:
                self._hits += 1
                # Return a shallow copy so caller can stamp cache_hit=True and fresh latency
                return {
                    "answer": best_entry.response_payload["answer"],
                    "tools": list(best_entry.response_payload.get("tools", [])),
                    "trace": dict(best_entry.response_payload.get("trace", {})),
                    "cache_similarity": round(best_sim, 4),
                }

            self._misses += 1
            return None

    def store(
        self,
        query: str,
        tenant_id: str,
        user_roles: List[str],
        retrieval_mode: str,
        enable_web_search: bool,
        response_payload: Dict[str, Any],
        doc_ids: Set[str],
    ) -> None:
        """Stores a completed RAG response along with its contributing doc_ids for future invalidation."""
        scope = self._make_scope(tenant_id, user_roles, retrieval_mode, enable_web_search)
        embedder = ModelRegistry.get_embeddings()
        q_vec = embedder.embed_query(query)

        with self._lock:
            if len(self._entries) >= self.max_entries:
                self._entries.pop(0)
            self._entries.append(
                SemanticCacheEntry(
                    cache_key_scope=scope,
                    query_text=query,
                    query_vector=q_vec,
                    response_payload={
                        "answer": response_payload["answer"],
                        "tools": list(response_payload.get("tools", [])),
                        "trace": dict(response_payload.get("trace", {})),
                    },
                    doc_ids=set(doc_ids),
                )
            )

    def invalidate_by_doc_id(self, doc_id: Optional[str]) -> int:
        """Evicts all cached answers that depended on a given doc_id."""
        if not doc_id:
            return 0
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if doc_id not in e.doc_ids]
            return before - len(self._entries)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cached_entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "similarity_threshold": self.similarity_threshold,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0


semantic_cache = SemanticQueryCache()
