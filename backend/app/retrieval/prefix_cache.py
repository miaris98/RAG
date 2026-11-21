"""#12 LinkedIn Shared-Prefix Prompt Caching & KV-Cache Token Amortization."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import threading
from typing import Any, Dict, List


CANONICAL_SYSTEM_PREFIX = """You are a Staff-level Production RAG Assistant operating under strict grounding and citation rules.

CRITICAL INSTRUCTIONS:
1. Structure your response with these exact three sections:
   ### According to internal documents:
   - If VERIFIED INTERNAL DOCUMENT CHUNKS is 'NO_RELEVANT_INTERNAL_DOCUMENTS_FOUND', state clearly: "My internal documents do not contain relevant information for this query."
   - Otherwise, answer using ONLY facts stated in the verified chunks and cite every claim inline using `[Doc 1]`, `[Doc 2]`, etc. Prioritize internal documents as authoritative truth even if they conflict with public internet results.

   ### According to the internet:
   - Summarize the external internet search findings concisely.

   ### Comparison:
   - Explicitly highlight any conflict, version difference, or agreement between the internal documents and the internet."""


class SharedPrefixPromptCache:
    """Orders system instructions + context before dynamic query text and tracks reusable KV prefix hits."""

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._prefixes: "OrderedDict[str, int]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._amortized_tokens_total = 0

    def build_prompt(
        self,
        context_blocks: List[str],
        user_query: str,
        web_results: str = "External web search was not executed.",
        system_prefix: str = CANONICAL_SYSTEM_PREFIX,
    ) -> Dict[str, Any]:
        """Assembles a KV-cache-friendly prompt (shared static prefix first, dynamic query suffix last)."""
        internal_context_str = (
            "\n\n---\n\n".join(context_blocks)
            if context_blocks
            else "NO_RELEVANT_INTERNAL_DOCUMENTS_FOUND"
        )
        shared_prefix = (
            f"{system_prefix}\n\n"
            f"VERIFIED INTERNAL DOCUMENT CHUNKS (Post-Hybrid RRF & Cross-Encoder Reranking):\n"
            f"{internal_context_str}"
        )
        dynamic_suffix = (
            f"\n\nEXTERNAL INTERNET SEARCH RESULTS:\n{web_results}\n\n"
            f"USER QUESTION:\n{user_query}\n"
        )
        prefix_hash = hashlib.sha256(shared_prefix.encode("utf-8")).hexdigest()[:16]
        prefix_tokens = max(1, int(len(shared_prefix.split()) * 1.3))

        with self._lock:
            if prefix_hash in self._prefixes:
                self._prefixes.move_to_end(prefix_hash)
                self._hits += 1
                amortized = self._prefixes[prefix_hash]
                self._amortized_tokens_total += amortized
                cache_hit = True
            else:
                self._prefixes[prefix_hash] = prefix_tokens
                if len(self._prefixes) > self.max_entries:
                    self._prefixes.popitem(last=False)
                self._misses += 1
                amortized = 0
                cache_hit = False

        return {
            "prefix_hash": prefix_hash,
            "prefix_tokens": prefix_tokens,
            "prefix_cache_hit": cache_hit,
            "amortized_tokens": amortized,
            "full_prompt": shared_prefix + dynamic_suffix,
        }

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "cached_prefixes": len(self._prefixes),
                "prefix_hits": self._hits,
                "prefix_misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "amortized_prefix_tokens_total": self._amortized_tokens_total,
            }

    def clear(self) -> None:
        with self._lock:
            self._prefixes.clear()
            self._hits = 0
            self._misses = 0
            self._amortized_tokens_total = 0


prefix_cache = SharedPrefixPromptCache()
