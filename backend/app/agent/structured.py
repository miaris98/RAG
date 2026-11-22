"""DoorDash Inverted RAG (Constrained Taxonomy Enum) & Ramp Rule-Grounded Structured JSON Engine."""

from __future__ import annotations

import math
import re
from typing import Dict, List

from app.guardrails.audit import audit_logger
from app.models import CitationItem, StructuredQueryRequest, StructuredQueryResponse
from app.retrieval.embeddings import ModelRegistry
from app.retrieval.entity_graph import extract_entities
from app.retrieval.hybrid import cosine_similarity, retrieval_engine
from app.retrieval.reranker import tokenize_content_words


RULE_SIGNAL_RE = re.compile(
    r"\b(?:must|shall|required|remediation|execute|contact|encrypted|retention|slo|threshold|"
    r"err_[a-z0-9_]+|pd-[a-z0-9-]+|aes-\d+-[a-z]+|\d+\s*(?:ms|milliseconds|seconds|days|%))\b",
    re.IGNORECASE,
)


def _humanize_label(label: str) -> str:
    return re.sub(r"[_\-/]+", " ", label).strip()


def _extract_verifiable_rules(chunks: list, query: str) -> List[str]:
    """Extracts concrete policy/remediation/constraint sentences directly from authorized retrieved chunks."""
    q_tokens = set(tokenize_content_words(query))
    scored_rules: List[tuple[float, str]] = []
    seen: set[str] = set()

    for chk in chunks:
        if not chk.is_relevant and chk.calibrated_prob < 0.25:
            continue
        # Remove bracketed contextual header lines before extracting rule sentences
        raw_lines = [
            ln.strip(" -*•")
            for ln in chk.parent_content.splitlines()
            if ln.strip() and not ln.strip().startswith("[") and not ln.strip().startswith("#")
        ]
        body = " ".join(raw_lines)
        for sent in re.split(r"(?<=[.!?])\s+", body):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            key = sent.lower()
            if key in seen:
                continue
            has_rule_signal = bool(RULE_SIGNAL_RE.search(sent))
            s_tokens = set(tokenize_content_words(sent))
            overlap = len(q_tokens & s_tokens)
            if has_rule_signal or overlap >= 2:
                seen.add(key)
                score = overlap + (1.5 if has_rule_signal else 0.0) + chk.calibrated_prob
                scored_rules.append((score, sent[:280]))

    scored_rules.sort(key=lambda x: x[0], reverse=True)
    return [rule for _, rule in scored_rules[:5]]


def execute_structured_taxonomy_query(req: StructuredQueryRequest) -> StructuredQueryResponse:
    """Runs RBAC-filtered retrieval, DoorDash Inverted RAG taxonomy scoring, and Ramp rule-grounded JSON output."""
    chunks = retrieval_engine.retrieve(
        queries=[req.text],
        mode=req.retrieval_mode,
        top_k=req.top_k,
        promote_parents=True,
        tenant_id=req.tenant_id,
        user_roles=req.user_roles,
    )

    relevant_chunks = [c for c in chunks if c.is_relevant] or chunks[:1]
    evidence_text = " ".join(c.parent_content[:500] for c in relevant_chunks[:3])
    combined_signal = f"{req.text}\n{evidence_text}"
    signal_tokens = set(tokenize_content_words(combined_signal))

    # DoorDash Inverted RAG: embed combined signal + every candidate taxonomy label and score jointly
    embedder = ModelRegistry.get_embeddings()
    label_texts = [_humanize_label(lbl) for lbl in req.allowed_labels]
    vectors = embedder.embed_documents([combined_signal] + label_texts)
    sig_vec = vectors[0]
    label_vecs = vectors[1:]

    raw_scores: Dict[str, float] = {}
    for lbl, lbl_text, l_vec in zip(req.allowed_labels, label_texts, label_vecs):
        dense_sim = max(0.0, cosine_similarity(sig_vec, l_vec))
        lbl_tokens = [t for t in tokenize_content_words(lbl_text) if len(t) >= 3]
        lex_overlap = (
            sum(1 for t in lbl_tokens if t in signal_tokens) / max(1, len(lbl_tokens))
            if lbl_tokens
            else 0.0
        )
        raw_scores[lbl] = round(0.55 * dense_sim + 0.45 * lex_overlap, 4)

    # Softmax temperature scaling for calibrated probability confidence over the constrained enum
    temp = 0.25
    max_raw = max(raw_scores.values()) if raw_scores else 0.0
    exps = {lbl: math.exp((val - max_raw) / temp) for lbl, val in raw_scores.items()}
    denom = sum(exps.values()) or 1.0
    probs = {lbl: round(val / denom, 4) for lbl, val in exps.items()}

    best_label = max(raw_scores.items(), key=lambda kv: kv[1])[0]
    confidence = probs[best_label]

    rules = _extract_verifiable_rules(relevant_chunks, req.text)
    entities = sorted(
        {ent for c in relevant_chunks for ent in (c.matched_entities or extract_entities(c.parent_content)[:5])}
    )[:12]

    citations: List[CitationItem] = []
    for idx, chk in enumerate(relevant_chunks, start=1):
        if not chk.is_relevant and chk.calibrated_prob < 0.25:
            continue
        quote_lines = [
            ln.strip()
            for ln in chk.parent_content.splitlines()
            if ln.strip() and not ln.strip().startswith("[") and not ln.strip().startswith("#")
        ]
        citations.append(
            CitationItem(
                ref_id=f"Doc {idx}",
                doc_id=chk.doc_id,
                chunk_id=chk.chunk_id,
                title=chk.title,
                source=chk.source,
                section=chk.section,
                calibrated_prob=chk.calibrated_prob,
                supporting_quote=(" ".join(quote_lines))[:240],
                matched_terms=chk.matched_terms,
                matched_entities=chk.matched_entities,
            )
        )

    rationale = (
        f"Constrained classification selected '{best_label}' (confidence={confidence:.2%}) "
        f"across {len(req.allowed_labels)} allowed taxonomy labels grounded in {len(citations)} RBAC-authorized citation(s)."
    )

    audit_entry = audit_logger.log_event(
        event_type="structured_taxonomy_query",
        actor_tenant=req.tenant_id,
        actor_roles=req.user_roles,
        doc_ids=[c.doc_id for c in citations],
        chunk_ids=[c.chunk_id for c in citations],
        summary=f"Structured query classified as '{best_label}' ({confidence:.2%}) with {len(rules)} rules",
        metadata={"classification_label": best_label, "confidence": confidence, "allowed_labels_count": len(req.allowed_labels)},
    )

    return StructuredQueryResponse(
        classification_label=best_label,
        confidence=confidence,
        candidate_label_scores=probs,
        rationale=rationale,
        verifiable_rules_matched=rules,
        extracted_entities=entities,
        citations=citations,
        audit_event_hash=audit_entry["event_hash"],
    )
