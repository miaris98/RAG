"""#14 Spotify Hierarchical Semantic IDs & #20 Airbnb Tenant/Partition Shard Routing."""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional


DOMAIN_LEXICON = {
    "payments": ("payment", "ledger", "raft", "quorum", "settlement", "aegispay", "checkout", "clearing", "transfer"),
    "security": ("aes", "encrypt", "hsm", "vault", "key", "rotation", "pii", "injection", "guardrail", "audit", "gdpr"),
    "governance": ("compliance", "sovereign", "president", "regulatory", "pagerduty", "emea", "policy", "charter"),
    "ml_rag": ("rag", "bm25", "milvus", "rerank", "embedding", "chunk", "retrieval", "rrf", "faithfulness", "hallucination"),
    "infrastructure": ("shard", "cluster", "latency", "gpu", "minio", "rabbitmq", "prometheus", "grafana", "replica"),
}


def compute_semantic_id(
    text: str,
    vector: List[float],
    entities: Optional[List[str]] = None,
) -> str:
    """Computes a 4-level hierarchical Semantic ID: sem.<domain>.<topic>.c<00-15>."""
    lower = (text or "").lower()
    best_domain = "general"
    best_hits = 0
    best_topic = "core"

    for domain, keywords in DOMAIN_LEXICON.items():
        matched = [kw for kw in keywords if kw in lower]
        if len(matched) > best_hits:
            best_hits = len(matched)
            best_domain = domain
            best_topic = matched[0]

    if best_topic == "core" and entities:
        slug = re.sub(r"[^a-z0-9]+", "", entities[0].lower())
        if slug:
            best_topic = slug[:16]

    # 4-bit LSH quantization over dense vector partitions (RQ-VAE-inspired deterministic bucket)
    bucket = 0
    if vector:
        n = len(vector)
        stride = max(1, n // 4)
        for bit_idx in range(4):
            start = bit_idx * stride
            end = n if bit_idx == 3 else min(n, start + stride)
            part_sum = sum(vector[start:end])
            if part_sum >= 0.0:
                bucket |= 1 << bit_idx

    return f"sem.{best_domain}.{best_topic}.c{bucket:02d}"


def compute_shard_id(tenant_id: str, doc_id: str, shards_per_tenant: int = 2) -> str:
    """Computes a deterministic Airbnb-style tenant partition shard ID."""
    safe_tenant = re.sub(r"[^a-z0-9_]+", "_", (tenant_id or "default").strip().lower()).strip("_") or "default"
    digest = hashlib.sha256(f"{safe_tenant}:{doc_id}".encode("utf-8")).digest()
    shard_num = digest[0] % max(1, shards_per_tenant)
    return f"shard_{safe_tenant}_{shard_num}"
