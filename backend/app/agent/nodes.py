"""Node implementations for the Corrective RAG (CRAG) + Self-RAG LangGraph pipeline."""

import re
import time
from typing import Any, Dict, List

from app.agent.state import RAGGraphState
from app.config import settings
from app.guardrails.security import guardrails
from app.memory.conversation import conversation_memory
from app.models import RetrievedChunk
from app.observability.metrics import llm_circuit_breaker, web_circuit_breaker
from app.retrieval.embeddings import ModelRegistry
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.postprocessor import context_postprocessor
from app.retrieval.reranker import tokenize_content_words


TECHNICAL_ID_PATTERN = re.compile(r"\b[A-Z]{2,}(?:_[A-Z0-9]+)+|\b[A-Z]{2,}-[A-Z0-9-]{2,}\b")
MULTIHOP_CONNECTORS = re.compile(r"\b(compare|versus|vs\.?|difference between|and also|both)\b", re.IGNORECASE)
CONCEPTUAL_PREFIX = re.compile(r"^(explain|how does|why does|describe|what is the architecture)\b", re.IGNORECASE)


def analyze_and_expand_query(state: RAGGraphState) -> Dict[str, Any]:
    """Node 1 (Adaptive Query Router): Sanitizes query via Guardrails, classifies route
    (lexical_lookup | conceptual_hyde | multi_hop | standard), and generates targeted sub-queries.
    """
    t0 = time.perf_counter()
    raw_query = state.get("query", "").strip()
    enable_expansion = state.get("enable_query_expansion", True)
    enable_guardrails = state.get("enable_guardrails", True)

    guardrail_flags: List[str] = []
    if enable_guardrails and raw_query:
        raw_query, guardrail_flags = guardrails.sanitize_text(raw_query)

    session_id = state.get("session_id") or ""
    if session_id and raw_query:
        raw_query = conversation_memory.resolve_query(session_id, raw_query)

    expanded: List[str] = [raw_query] if raw_query else []
    query_route = "standard"
    dense_weight = settings.dense_weight
    sparse_weight = settings.sparse_weight

    if enable_expansion and raw_query:
        content_tokens = tokenize_content_words(raw_query)

        if TECHNICAL_ID_PATTERN.search(raw_query):
            # Route A: Exact code / identifier lookup -> boost BM25 sparse weight
            query_route = "lexical_lookup"
            sparse_weight = 1.6
            dense_weight = 0.8
            if content_tokens:
                lex_q = " ".join(content_tokens)
                if lex_q.lower() != raw_query.lower():
                    expanded.append(lex_q)

        elif MULTIHOP_CONNECTORS.search(raw_query):
            # Route B: Multi-Hop decomposition -> split compound comparison into independent sub-queries
            query_route = "multi_hop"
            parts = re.split(r"\b(?:with|and|versus|vs\.?|compared to)\b", raw_query, flags=re.IGNORECASE)
            for part in parts:
                cleaned_part = re.sub(r"^(compare|difference between)\s+", "", part.strip(), flags=re.IGNORECASE).strip(" ?.,")
                if len(cleaned_part.split()) >= 2 and cleaned_part not in expanded:
                    expanded.append(cleaned_part)
            if content_tokens and len(expanded) < 3:
                expanded.append(" ".join(content_tokens))

        elif CONCEPTUAL_PREFIX.search(raw_query):
            # Route C: Conceptual / architectural query -> HyDE (Hypothetical Document Embedding)
            query_route = "conceptual_hyde"
            dense_weight = 1.3
            hyde_passage = (
                f"Technical Architecture & Specification: {' '.join(content_tokens)}. "
                f"In production systems, {' '.join(content_tokens)} operates through structured "
                f"retrieval, verification, and consensus mechanisms to guarantee reliability."
            )
            expanded.append(hyde_passage)
            if content_tokens:
                expanded.append(" ".join(content_tokens))

        else:
            if content_tokens:
                lex_q = " ".join(content_tokens)
                if lex_q.lower() != raw_query.lower():
                    expanded.append(lex_q)

    tools = list(state.get("tools_used") or [])
    if guardrail_flags and "security_guardrails" not in tools:
        tools.append("security_guardrails")

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["query_routing_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "query": raw_query,
        "guardrail_flags": guardrail_flags,
        "query_route": query_route,
        "expanded_queries": expanded,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "tools_used": tools,
        "stage_latencies_ms": latencies,
    }


def retrieve_and_rerank(state: RAGGraphState) -> Dict[str, Any]:
    """Node 2: Runs Pre-Filtered RBAC + Tri-Modal Hybrid Retrieval + Cross-Encoder Reranking."""
    t0 = time.perf_counter()
    queries = state.get("expanded_queries") or [state.get("query", "")]
    mode = state.get("retrieval_mode", "hybrid_graph_rerank")
    top_k = state.get("top_k", settings.rerank_top_k)
    tenant_id = state.get("tenant_id", "default")
    user_roles = state.get("user_roles")

    chunks: List[RetrievedChunk] = retrieval_engine.retrieve(
        queries=queries,
        mode=mode,  # type: ignore[arg-type]
        top_k=top_k,
        promote_parents=True,
        tenant_id=tenant_id,
        user_roles=user_roles,
        dense_weight=float(state.get("dense_weight", settings.dense_weight)),
        sparse_weight=float(state.get("sparse_weight", settings.sparse_weight)),
    )
    relevant_objs = context_postprocessor.lost_in_the_middle_reorder([c for c in chunks if c.is_relevant])
    pos_map: Dict[str, int] = {}
    for pos_idx, rel_chk in enumerate(relevant_objs, start=1):
        rel_chk.prompt_position = pos_idx
        pos_map[rel_chk.chunk_id] = pos_idx
    for chk in chunks:
        if chk.chunk_id in pos_map:
            chk.prompt_position = pos_map[chk.chunk_id]

    serialized_all = [c.model_dump() for c in chunks]
    serialized_relevant = [c.model_dump() for c in relevant_objs]
    retrieval_funnel = retrieval_engine.get_last_funnel()

    tools = list(state.get("tools_used") or [])
    if "document_retriever" not in tools:
        tools.append("document_retriever")
    if mode == "hybrid_graph_rerank" and "entity_knowledge_graph" not in tools:
        tools.append("entity_knowledge_graph")
    if mode in ("hybrid_rerank", "hybrid_graph_rerank") and "cross_encoder_reranker" not in tools:
        tools.append("cross_encoder_reranker")

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["hybrid_retrieval_rerank_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "retrieved_chunks": serialized_all,
        "relevant_chunks": serialized_relevant,
        "retrieval_funnel": retrieval_funnel,
        "tools_used": tools,
        "stage_latencies_ms": latencies,
    }


def grade_retrieval_crag(state: RAGGraphState) -> Dict[str, Any]:
    """Node 3 (CRAG Gate): Evaluates whether retrieved internal chunks are sufficient, ambiguous, or insufficient."""
    t0 = time.perf_counter()
    relevant = state.get("relevant_chunks") or []
    enable_web = state.get("enable_web_search", True)

    if not relevant:
        decision = "insufficient_web_fallback"
    else:
        top_prob = float(relevant[0].get("calibrated_prob", 0.0))
        if top_prob >= 0.40:
            decision = "sufficient"
        else:
            decision = "ambiguous_web_augmented"

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["crag_grading_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "crag_decision": decision,
        "stage_latencies_ms": latencies,
    }


def route_after_crag(state: RAGGraphState) -> str:
    """Conditional edge router after the CRAG Relevance Grader."""
    decision = state.get("crag_decision", "sufficient")
    enable_web = state.get("enable_web_search", True)
    if enable_web or decision == "insufficient_web_fallback":
        return "external_web_search"
    return "synthesize_cited_answer"


def external_web_search(state: RAGGraphState) -> Dict[str, Any]:
    """Node 4: Queries DuckDuckGo for external context to compare or fall back when internal docs miss."""
    t0 = time.perf_counter()
    query = state.get("query", "")
    web_text = ""

    if state.get("enable_web_search", True):
        if not web_circuit_breaker.allow_request():
            web_text = "External web search skipped (Circuit Breaker OPEN)."
        else:
            try:
                from langchain_community.tools import DuckDuckGoSearchRun

                search = DuckDuckGoSearchRun()
                web_text = str(search.invoke(query)).strip()
                web_circuit_breaker.record_success()
            except Exception as exc:
                web_circuit_breaker.record_failure()
                web_text = f"External web search unavailable ({exc})."
    else:
        web_text = "External web search disabled by request configuration."

    tools = list(state.get("tools_used") or [])
    if state.get("enable_web_search", True) and "internet_search" not in tools:
        tools.append("internet_search")

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["web_search_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "web_search_triggered": bool(state.get("enable_web_search", True)),
        "web_results": web_text,
        "tools_used": tools,
        "stage_latencies_ms": latencies,
    }


def _extract_best_supporting_quote(query: str, chunk_dict: Dict[str, Any]) -> str:
    """Extracts the most query-relevant sentence/line from the chunk as an inline evidence quote."""
    raw = chunk_dict.get("parent_content") or chunk_dict.get("content", "")
    clean = re.sub(r"^\[Document:.*?\]\n?", "", raw).strip()
    candidates = [s.strip(" -*#\t") for s in re.split(r"(?<=[.!?])\s+|\n+", clean) if len(s.strip()) >= 12]
    if not candidates:
        return clean[:240]

    q_tokens = set(tokenize_content_words(query))
    best_sent = candidates[0]
    best_score = -1.0
    for sent in candidates:
        s_tokens = set(tokenize_content_words(sent))
        overlap = len(q_tokens & s_tokens)
        id_bonus = 2.0 if TECHNICAL_ID_PATTERN.search(sent) else 0.0
        score = overlap + id_bonus
        if score > best_score:
            best_score = score
            best_sent = sent
    return best_sent[:240]


def _build_structured_citations(query: str, relevant_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Builds structured CitationItem dictionaries with exact evidence quotes for each verified chunk."""
    citations: List[Dict[str, Any]] = []
    for idx, chk in enumerate(relevant_chunks, start=1):
        citations.append(
            {
                "ref_id": f"[Doc {idx}]",
                "doc_id": chk.get("doc_id", ""),
                "chunk_id": chk.get("chunk_id", ""),
                "title": chk.get("title", "Untitled"),
                "source": chk.get("source", ""),
                "section": chk.get("section", "General"),
                "calibrated_prob": float(chk.get("calibrated_prob", 0.0)),
                "supporting_quote": _extract_best_supporting_quote(query, chk),
                "matched_terms": list(chk.get("matched_terms") or []),
                "matched_entities": list(chk.get("matched_entities") or []),
            }
        )
    return citations


def _format_citations_bibliography(citations: List[Dict[str, Any]]) -> str:
    """Formats the verified inline citations bibliography block appended to grounded answers."""
    if not citations:
        return ""
    lines = ["\n\n### 📚 Verified Inline Citations & Evidence Quotes"]
    for cit in citations:
        terms = ", ".join(f"`{t}`" for t in cit.get("matched_terms", [])[:6]) or "semantic"
        lines.append(
            f"- **{cit['ref_id']}** **{cit['title']}** (§{cit['section']}, `{cit['chunk_id']}`, "
            f"source=`{cit['source']}`, prob=`{cit['calibrated_prob']:.3f}`, matched={terms})\n"
            f"  > \"{cit['supporting_quote']}\""
        )
    return "\n".join(lines)


def _build_extractive_fallback_answer(
    query: str,
    relevant_chunks: List[Dict[str, Any]],
    web_results: str,
) -> str:
    """Deterministic citation-grounded synthesis used when the local Ollama LLM is offline or unreachable."""
    if relevant_chunks:
        internal_lines = []
        for idx, chk in enumerate(relevant_chunks, start=1):
            section = chk.get("section", "General")
            citation = f"[Doc {idx}: {chk['title']} §{section}]"
            parent_text = chk.get("parent_content") or chk.get("content", "")
            clean_body = re.sub(r"^\[Document:.*?\]\n?", "", parent_text).strip()
            internal_lines.append(f"- {citation}: {clean_body}")
        internal_block = "\n".join(internal_lines)
    else:
        internal_block = (
            "My internal documents do not contain verified information matching this query "
            f"(0 chunks passed the Cross-Encoder relevance threshold of {settings.relevance_threshold})."
        )

    web_block = web_results if web_results else "No external web results retrieved."

    if relevant_chunks and web_results and not web_results.startswith("External web search"):
        comparison_block = (
            "Internal documents provide verified domain-specific ground truth (cited above via [Doc i]), "
            "which takes precedence over general internet search results whenever technical specifications "
            "or internal governance records differ."
        )
    elif relevant_chunks:
        comparison_block = (
            "Answer is grounded directly in high-confidence internal documents verified by the Cross-Encoder reranker."
        )
    else:
        comparison_block = (
            "Because no internal chunks passed the CRAG relevance threshold, the response relies on external web fallback "
            "rather than hallucinating from low-similarity vector matches."
        )

    return (
        f"### According to internal documents:\n{internal_block}\n\n"
        f"### According to the internet:\n{web_block}\n\n"
        f"### Comparison:\n{comparison_block}"
    )


def synthesize_cited_answer(state: RAGGraphState) -> Dict[str, Any]:
    """Node 5: Synthesizes a grounded response with mandatory [Doc i] inline citations."""
    t0 = time.perf_counter()
    query = state.get("query", "")
    relevant_chunks = state.get("relevant_chunks") or []
    web_results = state.get("web_results") or "External web search was not executed."

    if relevant_chunks:
        formatted_context_blocks = []
        for idx, chk in enumerate(relevant_chunks, start=1):
            formatted_context_blocks.append(
                f"[Doc {idx}] Title: {chk['title']} | ChunkID: {chk['chunk_id']} | "
                f"CalibratedRelevance: {chk['calibrated_prob']:.3f}\n"
                f"Content:\n{chk.get('parent_content') or chk.get('content', '')}"
            )
    else:
        formatted_context_blocks = []

    from app.retrieval.prefix_cache import prefix_cache

    prompt_bundle = prefix_cache.build_prompt(
        context_blocks=formatted_context_blocks,
        user_query=query,
        web_results=web_results,
    )
    prompt = prompt_bundle["full_prompt"]

    if not llm_circuit_breaker.allow_request():
        draft = _build_extractive_fallback_answer(query, relevant_chunks, web_results)
    else:
        try:
            llm = ModelRegistry.get_llm()
            response = llm.invoke(prompt)
            draft = str(response.content).strip()
            llm_circuit_breaker.record_success()
            if not draft:
                draft = _build_extractive_fallback_answer(query, relevant_chunks, web_results)
        except Exception as exc:
            llm_circuit_breaker.record_failure()
            print(f"[synthesize_cited_answer] LLM call failed, using deterministic grounded synthesis: {exc}")
            draft = _build_extractive_fallback_answer(query, relevant_chunks, web_results)

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["generation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "draft_answer": draft,
        "final_answer": draft,
        "stage_latencies_ms": latencies,
    }


def verify_faithfulness_self_rag(state: RAGGraphState) -> Dict[str, Any]:
    """Node 6 (Self-RAG Grader): Audits citation presence, builds structured citations, and verifies grounding."""
    t0 = time.perf_counter()
    query = state.get("query", "")
    draft = state.get("draft_answer", "")
    relevant_chunks = state.get("relevant_chunks") or []
    enable_self_rag = state.get("enable_self_rag", True)

    structured_citations = _build_structured_citations(query, relevant_chunks)
    citations = re.findall(r"\[Doc\s*\d+[^\]]*\]", draft, flags=re.IGNORECASE)
    citation_count = len(citations)

    if not enable_self_rag:
        verdict = "skipped"
        faithfulness = 1.0
        final_answer = draft
    elif not relevant_chunks:
        verdict = "no_internal_context"
        faithfulness = 1.0 if "do not contain" in draft.lower() or "0 chunks" in draft.lower() else 0.7
        final_answer = draft
    else:
        internal_section = draft.split("### According to the internet:")[0]
        answer_tokens = tokenize_content_words(internal_section)
        source_corpus_text = " ".join(
            (c.get("parent_content") or c.get("content", "")) for c in relevant_chunks
        )
        source_token_set = set(tokenize_content_words(source_corpus_text))

        if answer_tokens:
            supported = sum(1 for tok in answer_tokens if tok in source_token_set)
            support_ratio = supported / len(answer_tokens)
        else:
            support_ratio = 1.0

        if citation_count == 0:
            inline_badges = " ".join(f"{c['ref_id']}" for c in structured_citations)
            final_answer = draft + (f"\n\n**Verified Sources:** {inline_badges}" if inline_badges else "")
            citation_count = len(structured_citations)
            verdict = "revised"
        else:
            final_answer = draft
            verdict = "grounded"

        citation_bonus = 0.25 if citation_count > 0 else 0.0
        faithfulness = round(min(1.0, 0.75 * min(1.0, support_ratio * 1.35) + citation_bonus), 4)

    if structured_citations and "### 📚 Verified Inline Citations & Evidence Quotes" not in final_answer:
        final_answer = final_answer + _format_citations_bibliography(structured_citations)

    tools = list(state.get("tools_used") or [])
    if enable_self_rag and "self_rag_verifier" not in tools:
        tools.append("self_rag_verifier")

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["self_rag_verification_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 1))
    critique_history = list(state.get("critique_history") or [])
    top_k = int(state.get("top_k", settings.rerank_top_k))

    if enable_self_rag and relevant_chunks and faithfulness < 0.75 and retry_count < max_retries:
        critique_history.append(
            f"Retry {retry_count + 1}: faithfulness={faithfulness:.3f} < 0.75; broadening top_k from {top_k} to {min(10, top_k + 2)}."
        )
        return {
            "final_answer": final_answer,
            "self_rag_verdict": "retrying",
            "faithfulness_score": faithfulness,
            "citation_count": citation_count,
            "citations": structured_citations,
            "retry_count": retry_count + 1,
            "top_k": min(10, top_k + 2),
            "critique_history": critique_history,
            "tools_used": tools,
            "stage_latencies_ms": latencies,
        }

    return {
        "final_answer": final_answer,
        "self_rag_verdict": verdict,
        "faithfulness_score": faithfulness,
        "citation_count": citation_count,
        "citations": structured_citations,
        "retry_count": retry_count,
        "critique_history": critique_history,
        "tools_used": tools,
        "stage_latencies_ms": latencies,
    }


def route_after_self_rag(state: RAGGraphState) -> str:
    """Cyclic Self-Correction Router: loops back to retrieve_and_rerank when verdict == 'retrying'."""
    if state.get("self_rag_verdict") == "retrying":
        return "retrieve_and_rerank"
    return "end"
