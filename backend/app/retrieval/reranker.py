"""Cross-Encoder Reranker with Sigmoid Score Calibration and CRAG Relevance Gating.

Why Cross-Encoder Reranking matters in Staff RAG Interviews:
- Bi-Encoders compress query and passage independently into fixed-size vectors (e.g. 384-d),
  losing fine-grained token-to-token interaction (negations, exact numbers, multi-constraint logic).
- A Cross-Encoder passes [CLS] query [SEP] passage [SEP] through full Transformer self-attention,
  computing joint cross-attention across every query and document token.
- Calibrating raw MS-MARCO logits via a temperature-scaled sigmoid provides an interpretable
  [0, 1] probability used by the Corrective RAG (CRAG) gate to filter out irrelevant top-k noise.
"""

import math
import re
from typing import List, Set

from app.config import settings
from app.models import RetrievedChunk
from app.retrieval.embeddings import ModelRegistry


STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "about", "as",
    "into", "like", "through", "after", "over", "between", "out", "against",
    "during", "without", "before", "under", "around", "among", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "have", "has", "had",
    "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "and", "or", "but", "if", "while", "how", "when", "where", "why",
}


def tokenize_content_words(text: str) -> List[str]:
    """Extracts normalized alphanumeric + technical hyphenated/dotted tokens excluding stopwords."""
    raw_tokens = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())
    return [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]


def sigmoid(logit: float, temperature: float = 1.5) -> float:
    """Maps unbounded Cross-Encoder logits to calibrated [0.0, 1.0] probabilities."""
    scaled = max(-20.0, min(20.0, logit / temperature))
    return 1.0 / (1.0 + math.exp(-scaled))


class CrossEncoderReranker:
    """Scores candidate chunks with a Cross-Encoder and applies CRAG relevance gating."""

    def __init__(self, relevance_threshold: float = settings.relevance_threshold):
        self.relevance_threshold = relevance_threshold

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int = settings.rerank_top_k,
    ) -> List[RetrievedChunk]:
        """Reranks candidate chunks jointly against the query and calibrates relevance probabilities."""
        if not candidates:
            return []

        cross_encoder = ModelRegistry.get_reranker()
        pairs = [(query, c.content) for c in candidates]

        if cross_encoder is not None:
            try:
                raw_scores = cross_encoder.predict(pairs)
                for chunk, raw_logit in zip(candidates, raw_scores):
                    logit_val = float(raw_logit)
                    # Blend cross-encoder logit probability with lexical anchor bonus for exact identifiers
                    ce_prob = sigmoid(logit_val, temperature=1.5)
                    lex_overlap = self._lexical_overlap_score(query, chunk.content)
                    calibrated = min(0.999, 0.82 * ce_prob + 0.18 * lex_overlap)
                    chunk.rerank_logit = round(logit_val, 4)
                    chunk.calibrated_prob = round(calibrated, 4)
                    chunk.is_relevant = chunk.calibrated_prob >= self.relevance_threshold
            except Exception as exc:
                print(f"[CrossEncoderReranker] Fallback scoring due to error: {exc}")
                self._apply_fallback_scores(query, candidates)
        else:
            self._apply_fallback_scores(query, candidates)

        # Sort descending by calibrated_prob (breaking ties by rrf_score)
        ranked = sorted(
            candidates,
            key=lambda c: (c.calibrated_prob, c.rrf_score),
            reverse=True,
        )
        return ranked[:top_k]

    def _lexical_overlap_score(self, query: str, doc_text: str) -> float:
        """Computes content-word recall + bigram phrase bonus in [0.0, 1.0]."""
        q_tokens = tokenize_content_words(query)
        if not q_tokens:
            return 0.0
        d_tokens = tokenize_content_words(doc_text)
        if not d_tokens:
            return 0.0

        d_set = set(d_tokens)
        matched = sum(1 for t in q_tokens if t in d_set)
        unigram_recall = matched / len(q_tokens)

        # Bigram exact ordering bonus
        if len(q_tokens) >= 2 and len(d_tokens) >= 2:
            q_bigrams = set(zip(q_tokens, q_tokens[1:]))
            d_bigrams = set(zip(d_tokens, d_tokens[1:]))
            bigram_recall = len(q_bigrams & d_bigrams) / len(q_bigrams)
        else:
            bigram_recall = unigram_recall

        return min(1.0, 0.7 * unigram_recall + 0.3 * bigram_recall)

    def _apply_fallback_scores(self, query: str, candidates: List[RetrievedChunk]) -> None:
        """Deterministic cross-scoring when running offline without CrossEncoder weights."""
        for chunk in candidates:
            lex = self._lexical_overlap_score(query, chunk.content)
            dense_component = max(0.0, min(1.0, chunk.dense_score or 0.0))
            rrf_component = min(1.0, chunk.rrf_score * 30.0)
            prob = min(0.99, 0.55 * lex + 0.25 * dense_component + 0.20 * rrf_component)
            logit = math.log(max(1e-4, prob) / max(1e-4, 1.0 - prob))
            chunk.rerank_logit = round(logit, 4)
            chunk.calibrated_prob = round(prob, 4)
            chunk.is_relevant = chunk.calibrated_prob >= self.relevance_threshold
