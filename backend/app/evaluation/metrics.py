"""Standard Information Retrieval (IR) and RAG evaluation metrics:
- HitRate@K (Recall@K binary hit)
- MRR@K (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- ContextPrecision@K (Fraction of top-K retrieved chunks that match ground-truth evidence)
"""

import math
from typing import List


def is_chunk_match(chunk_text: str, expected_keywords: List[str]) -> bool:
    """Determines whether a retrieved chunk contains the ground-truth answer evidence."""
    lowered = chunk_text.lower()
    # A chunk is a ground-truth positive if it contains all required anchor phrases/keywords
    return all(kw.lower() in lowered for kw in expected_keywords)


def compute_relevance_vector(retrieved_texts: List[str], expected_keywords: List[str], k: int) -> List[int]:
    """Produces a binary relevance list [rel_1, rel_2, ..., rel_k] for top-k retrieved chunks."""
    top_slice = retrieved_texts[:k]
    return [1 if is_chunk_match(txt, expected_keywords) else 0 for txt in top_slice]


def hit_rate_at_k(rel_vector: List[int]) -> float:
    """Returns 1.0 if at least one relevant chunk appears in top-K, else 0.0."""
    return 1.0 if any(r > 0 for r in rel_vector) else 0.0


def reciprocal_rank_at_k(rel_vector: List[int]) -> float:
    """Returns 1 / rank of the first relevant chunk in top-K, or 0.0 if none found."""
    for idx, rel in enumerate(rel_vector, start=1):
        if rel > 0:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(rel_vector: List[int]) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG@K) for binary relevance."""
    if not rel_vector or not any(r > 0 for r in rel_vector):
        return 0.0
    dcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(rel_vector, start=1))
    ideal_rels = sorted(rel_vector, reverse=True)
    idcg = sum(rel / math.log2(idx + 1) for idx, rel in enumerate(ideal_rels, start=1))
    return (dcg / idcg) if idcg > 0 else 0.0


def context_precision_at_k(rel_vector: List[int]) -> float:
    """Computes precision@K: fraction of retrieved top-K chunks that are relevant."""
    if not rel_vector:
        return 0.0
    return sum(rel_vector) / len(rel_vector)
