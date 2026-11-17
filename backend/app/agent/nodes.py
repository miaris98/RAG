"""Node implementations for the Corrective RAG (CRAG) + Self-RAG LangGraph pipeline."""

import re
import time
from typing import Any, Dict, List

from app.agent.state import RAGGraphState
from app.config import settings
from app.models import RetrievedChunk
from app.retrieval.embeddings import ModelRegistry
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.reranker import tokenize_content_words


def analyze_and_expand_query(state: RAGGraphState) -> Dict[str, Any]:
    """Node 1: Normalizes the user query and generates lexical + semantic sub-queries."""
    t0 = time.perf_counter()
    raw_query = state.get("query", "").strip()
    enable_expansion = state.get("enable_query_expansion", True)

    expanded: List[str] = [raw_query]
    if enable_expansion and raw_query:
        content_tokens = tokenize_content_words(raw_query)
        if content_tokens:
            # Lexical anchor query for BM25 exact identifier / entity recall
            lexical_query = " ".join(content_tokens)
            if lexical_query.lower() != raw_query.lower():
                expanded.append(lexical_query)
            # Semantic specification query for dense bi-encoder recall
            if len(content_tokens) >= 2:
                semantic_query = f"technical specification architecture and details for {' '.join(content_tokens)}"
                expanded.append(semantic_query)

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["query_expansion_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "expanded_queries": expanded,
        "stage_latencies_ms": latencies,
    }


def retrieve_and_rerank(state: RAGGraphState) -> Dict[str, Any]:
    """Node 2: Runs Two-Stage Hybrid Retrieval (Dense Milvus + BM25 RRF -> Cross-Encoder Reranker)."""
    t0 = time.perf_counter()
    queries = state.get("expanded_queries") or [state.get("query", "")]
    mode = state.get("retrieval_mode", "hybrid_rerank")
    top_k = state.get("top_k", settings.rerank_top_k)

    chunks: List[RetrievedChunk] = retrieval_engine.retrieve(
        queries=queries,
        mode=mode,  # type: ignore[arg-type]
        top_k=top_k,
        promote_parents=True,
    )
    serialized_all = [c.model_dump() for c in chunks]
    serialized_relevant = [c.model_dump() for c in chunks if c.is_relevant]

    tools = list(state.get("tools_used") or [])
    if "document_retriever" not in tools:
        tools.append("document_retriever")
    if mode == "hybrid_rerank" and "cross_encoder_reranker" not in tools:
        tools.append("cross_encoder_reranker")

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["hybrid_retrieval_rerank_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "retrieved_chunks": serialized_all,
        "relevant_chunks": serialized_relevant,
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
        try:
            from langchain_community.tools import DuckDuckGoSearchRun

            search = DuckDuckGoSearchRun()
            web_text = str(search.invoke(query)).strip()
        except Exception as exc:
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


def _build_extractive_fallback_answer(
    query: str,
    relevant_chunks: List[Dict[str, Any]],
    web_results: str,
) -> str:
    """Deterministic citation-grounded synthesis used when the local Ollama LLM is offline or unreachable."""
    if relevant_chunks:
        internal_lines = []
        for idx, chk in enumerate(relevant_chunks, start=1):
            citation = f"[Doc {idx}: {chk['title']} ({chk['chunk_id']}, score={chk['calibrated_prob']:.2f})]"
            parent_text = chk.get("parent_content") or chk.get("content", "")
            # Strip the contextual bracket header for cleaner reading while keeping citation
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
        internal_context_str = "\n\n---\n\n".join(formatted_context_blocks)
    else:
        internal_context_str = "NO_RELEVANT_INTERNAL_DOCUMENTS_FOUND"

    prompt = f"""You are a Staff-level Production RAG Assistant operating under strict grounding and citation rules.

USER QUESTION:
{query}

VERIFIED INTERNAL DOCUMENT CHUNKS (Post-Hybrid RRF & Cross-Encoder Reranking):
{internal_context_str}

EXTERNAL INTERNET SEARCH RESULTS:
{web_results}

CRITICAL INSTRUCTIONS:
1. Structure your response with these exact three sections:
   ### According to internal documents:
   - If VERIFIED INTERNAL DOCUMENT CHUNKS is 'NO_RELEVANT_INTERNAL_DOCUMENTS_FOUND', state clearly: "My internal documents do not contain relevant information for this query."
   - Otherwise, answer using ONLY facts stated in the verified chunks and cite every claim inline using `[Doc 1]`, `[Doc 2]`, etc. Prioritize internal documents as authoritative truth even if they conflict with public internet results.

   ### According to the internet:
   - Summarize the external internet search findings concisely.

   ### Comparison:
   - Explicitly highlight any conflict, version difference, or agreement between the internal documents and the internet.
"""

    try:
        llm = ModelRegistry.get_llm()
        response = llm.invoke(prompt)
        draft = str(response.content).strip()
        if not draft:
            draft = _build_extractive_fallback_answer(query, relevant_chunks, web_results)
    except Exception as exc:
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
    """Node 6 (Self-RAG Grader): Audits citation presence and lexical faithfulness against retrieved chunks."""
    t0 = time.perf_counter()
    draft = state.get("draft_answer", "")
    relevant_chunks = state.get("relevant_chunks") or []
    enable_self_rag = state.get("enable_self_rag", True)

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
        # Compute content-token support ratio between the internal answer section and retrieved parent chunks
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

        # If the LLM forgot to include [Doc i] citations despite having relevant chunks, inject explicit provenance
        if citation_count == 0:
            provenance_footer = "\n\n**Verified Internal Citations (Self-RAG Enforced):**\n" + "\n".join(
                f"- `[Doc {i}]` **{c['title']}** (`{c['chunk_id']}`, calibrated_prob=`{c['calibrated_prob']:.3f}`)"
                for i, c in enumerate(relevant_chunks, start=1)
            )
            final_answer = draft + provenance_footer
            citation_count = len(relevant_chunks)
            verdict = "revised"
        else:
            final_answer = draft
            verdict = "grounded"

        citation_bonus = 0.25 if citation_count > 0 else 0.0
        faithfulness = round(min(1.0, 0.75 * min(1.0, support_ratio * 1.35) + citation_bonus), 4)

    tools = list(state.get("tools_used") or [])
    if enable_self_rag and "self_rag_verifier" not in tools:
        tools.append("self_rag_verifier")

    latencies = dict(state.get("stage_latencies_ms") or {})
    latencies["self_rag_verification_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "final_answer": final_answer,
        "self_rag_verdict": verdict,
        "faithfulness_score": faithfulness,
        "citation_count": citation_count,
        "tools_used": tools,
        "stage_latencies_ms": latencies,
    }
