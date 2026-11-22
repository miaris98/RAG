"""Post-Retrieval Context Optimization:
1. Maximal Marginal Relevance (MMR) Diversity Selection (eliminates redundant chunks).
2. Lost-in-the-Middle U-Shaped Reordering (places highest-relevance chunks at start & end of context window).
3. Negation-Aware Consistency Check (detects polarity inversions like 'not', 'never', 'disabled').
"""

from typing import List, Set

from app.models import RetrievedChunk
from app.retrieval.reranker import tokenize_content_words


NEGATION_WORDS: Set[str] = {"not", "no", "never", "neither", "nor", "disabled", "cannot", "without"}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    toks_a = set(tokenize_content_words(text_a))
    toks_b = set(tokenize_content_words(text_b))
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


class ContextPostProcessor:
    """Applies MMR diversity and Lost-in-the-Middle U-shaped ordering."""

    def __init__(self, mmr_lambda: float = 0.65):
        self.mmr_lambda = mmr_lambda

    def select_mmr(self, candidates: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Selects diverse, high-relevance chunks using Maximal Marginal Relevance."""
        if len(candidates) <= top_k:
            return list(candidates)

        remaining = list(candidates)
        selected: List[RetrievedChunk] = [remaining.pop(0)]

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_mmr = -1e9
            for idx, cand in enumerate(remaining):
                rel = cand.calibrated_prob
                max_sim = max(
                    jaccard_similarity(cand.parent_content, sel.parent_content) for sel in selected
                )
                mmr_score = self.mmr_lambda * rel - (1.0 - self.mmr_lambda) * max_sim
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    def lost_in_the_middle_reorder(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Orders chunks in U-shaped attention layout: [Rank 1, Rank 3, ..., Rank 4, Rank 2]."""
        if len(chunks) <= 2:
            return list(chunks)
        left: List[RetrievedChunk] = []
        right: List[RetrievedChunk] = []
        for idx, chk in enumerate(chunks):
            if idx % 2 == 0:
                left.append(chk)
            else:
                right.append(chk)
        return left + list(reversed(right))

    @staticmethod
    def apply_negation_adjustment(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Penalizes chunks whose negation polarity contradicts an explicit negation in the query."""
        q_words = set(query.lower().split())
        q_has_neg = bool(q_words & NEGATION_WORDS)
        if not q_has_neg:
            return chunks
        for chk in chunks:
            c_words = set(chk.content.lower().split())
            c_has_neg = bool(c_words & NEGATION_WORDS)
            if not c_has_neg:
                chk.calibrated_prob = round(max(0.0, chk.calibrated_prob * 0.85), 4)
        return sorted(chunks, key=lambda c: c.calibrated_prob, reverse=True)


context_postprocessor = ContextPostProcessor()
