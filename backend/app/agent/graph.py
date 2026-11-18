"""Compiles, executes, and streams the cyclic LangGraph Corrective RAG (CRAG) + Self-RAG StateGraph."""

import json
import time
from typing import Any, Callable, Dict, Iterator

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = "__end__"

    class _CompiledStateGraph:
        def __init__(self, entry: str, nodes: dict, edges: dict, cond_edges: dict):
            self.entry = entry
            self.nodes = nodes
            self.edges = edges
            self.cond_edges = cond_edges

        def invoke(self, initial_state: dict) -> dict:
            state = dict(initial_state)
            curr = self.entry
            steps = 0
            while curr and curr != END and steps < 25:
                steps += 1
                node_fn = self.nodes[curr]
                updates = node_fn(state)
                if updates:
                    state.update(updates)
                if curr in self.cond_edges:
                    router_fn, mapping = self.cond_edges[curr]
                    route_key = router_fn(state)
                    curr = mapping.get(route_key, END)
                else:
                    curr = self.edges.get(curr, END)
            return state

        def stream(self, initial_state: dict) -> Iterator[Dict[str, Any]]:
            state = dict(initial_state)
            curr = self.entry
            steps = 0
            while curr and curr != END and steps < 25:
                steps += 1
                node_fn = self.nodes[curr]
                updates = node_fn(state)
                if updates:
                    state.update(updates)
                yield {curr: updates}
                if curr in self.cond_edges:
                    router_fn, mapping = self.cond_edges[curr]
                    route_key = router_fn(state)
                    curr = mapping.get(route_key, END)
                else:
                    curr = self.edges.get(curr, END)

    class StateGraph:  # type: ignore[no-redef]
        """Deterministic StateGraph fallback when langgraph package is not installed on host."""

        def __init__(self, state_schema: Any):
            self.state_schema = state_schema
            self._nodes: Dict[str, Callable] = {}
            self._edges: Dict[str, str] = {}
            self._cond_edges: Dict[str, Any] = {}
            self._entry: str = ""

        def add_node(self, name: str, fn: Callable) -> None:
            self._nodes[name] = fn

        def set_entry_point(self, name: str) -> None:
            self._entry = name

        def add_edge(self, src: str, dst: str) -> None:
            self._edges[src] = dst

        def add_conditional_edges(self, src: str, router_fn: Callable, mapping: Dict[str, str]) -> None:
            self._cond_edges[src] = (router_fn, mapping)

        def compile(self) -> _CompiledStateGraph:
            return _CompiledStateGraph(self._entry, self._nodes, self._edges, self._cond_edges)


from app.agent.nodes import (
    analyze_and_expand_query,
    external_web_search,
    grade_retrieval_crag,
    retrieve_and_rerank,
    route_after_crag,
    route_after_self_rag,
    synthesize_cited_answer,
    verify_faithfulness_self_rag,
)
from app.agent.state import RAGGraphState
from app.guardrails.audit import audit_logger
from app.memory.conversation import conversation_memory
from app.models import PipelineTrace, QueryInput, RetrievedChunk
from app.observability.metrics import metrics_collector
from app.observability.tracing import trace_collector
from app.retrieval.labeled_queries import labeled_query_store
from app.retrieval.semantic_cache import semantic_cache


def build_rag_state_graph():
    """Constructs the cyclic LangGraph StateGraph for CRAG + Self-RAG orchestration."""
    workflow = StateGraph(RAGGraphState)

    workflow.add_node("analyze_and_expand_query", analyze_and_expand_query)
    workflow.add_node("retrieve_and_rerank", retrieve_and_rerank)
    workflow.add_node("grade_retrieval_crag", grade_retrieval_crag)
    workflow.add_node("external_web_search", external_web_search)
    workflow.add_node("synthesize_cited_answer", synthesize_cited_answer)
    workflow.add_node("verify_faithfulness_self_rag", verify_faithfulness_self_rag)

    workflow.set_entry_point("analyze_and_expand_query")
    workflow.add_edge("analyze_and_expand_query", "retrieve_and_rerank")
    workflow.add_edge("retrieve_and_rerank", "grade_retrieval_crag")
    workflow.add_conditional_edges(
        "grade_retrieval_crag",
        route_after_crag,
        {
            "external_web_search": "external_web_search",
            "synthesize_cited_answer": "synthesize_cited_answer",
        },
    )
    workflow.add_edge("external_web_search", "synthesize_cited_answer")
    workflow.add_edge("synthesize_cited_answer", "verify_faithfulness_self_rag")
    # Cyclic Self-Correction loop back to retrieve_and_rerank if faithfulness is below threshold
    workflow.add_conditional_edges(
        "verify_faithfulness_self_rag",
        route_after_self_rag,
        {
            "retrieve_and_rerank": "retrieve_and_rerank",
            "end": END,
        },
    )

    return workflow.compile()


rag_compiled_graph = build_rag_state_graph()


def _build_initial_state(query_input: QueryInput) -> RAGGraphState:
    return {
        "query": query_input.text,
        "retrieval_mode": query_input.retrieval_mode,
        "top_k": query_input.top_k,
        "tenant_id": query_input.tenant_id,
        "user_roles": query_input.user_roles,
        "enable_query_expansion": query_input.enable_query_expansion,
        "enable_web_search": query_input.enable_web_search,
        "enable_self_rag": query_input.enable_self_rag,
        "enable_guardrails": query_input.enable_guardrails,
        "session_id": query_input.session_id or "",
        "max_retries": query_input.max_retries,
        "retry_count": 0,
        "critique_history": [],
        "tools_used": [],
        "stage_latencies_ms": {},
    }


def _package_output(query_input: QueryInput, final_state: RAGGraphState, total_ms: float) -> Dict[str, Any]:
    retrieved_raw = final_state.get("retrieved_chunks") or []
    relevant_raw = final_state.get("relevant_chunks") or []
    retrieved_models = [RetrievedChunk(**c) for c in retrieved_raw]

    trace = PipelineTrace(
        original_query=query_input.text,
        query_route=final_state.get("query_route", "standard"),
        expanded_queries=final_state.get("expanded_queries") or [query_input.text],
        retrieval_mode=query_input.retrieval_mode,
        tenant_id=query_input.tenant_id,
        user_roles=list(query_input.user_roles),
        cache_hit=False,
        guardrail_flags=final_state.get("guardrail_flags") or [],
        candidates_retrieved_count=len(retrieved_models),
        relevant_chunks_count=len(relevant_raw),
        crag_decision=final_state.get("crag_decision", "sufficient"),
        self_rag_verdict=final_state.get("self_rag_verdict", "grounded"),
        faithfulness_score=float(final_state.get("faithfulness_score", 1.0)),
        citation_count=int(final_state.get("citation_count", 0)),
        retry_count=int(final_state.get("retry_count", 0)),
        critique_history=list(final_state.get("critique_history") or []),
        session_id=query_input.session_id,
        stage_latencies_ms=final_state.get("stage_latencies_ms") or {},
        total_latency_ms=total_ms,
        retrieved_chunks=retrieved_models,
        web_search_triggered=bool(final_state.get("web_search_triggered", False)),
        web_snippets_preview=(final_state.get("web_results") or "")[:350],
    )

    return {
        "answer": final_state.get("final_answer", ""),
        "tools": final_state.get("tools_used") or [],
        "trace": trace.model_dump(),
    }


def _record_observability_and_audit(query_input: QueryInput, output: Dict[str, Any]) -> None:
    trace = output["trace"]
    metrics_collector.record_query(
        latency_ms=trace["total_latency_ms"],
        cache_hit=trace["cache_hit"],
        crag_decision=trace["crag_decision"],
        retry_count=trace.get("retry_count", 0),
        guardrail_flags=trace.get("guardrail_flags", []),
    )
    if query_input.session_id:
        conversation_memory.record_turn(query_input.session_id, query_input.text, output["answer"])

    doc_ids = sorted({c["doc_id"] for c in trace.get("retrieved_chunks", []) if c.get("doc_id")})
    chunk_ids = [c["chunk_id"] for c in trace.get("retrieved_chunks", []) if c.get("chunk_id")]
    trace_id = trace_collector.record_trace(query_input, output)
    trace["trace_id"] = trace_id
    audit_logger.log_event(
        event_type="rag_query",
        actor_tenant=query_input.tenant_id,
        actor_roles=query_input.user_roles,
        doc_ids=doc_ids,
        chunk_ids=chunk_ids,
        summary=f"Query='{query_input.text[:120]}' route={trace['query_route']} cache_hit={trace['cache_hit']}",
        metadata={
            "trace_id": trace_id,
            "crag_decision": trace["crag_decision"],
            "faithfulness_score": trace["faithfulness_score"],
            "guardrail_flags": trace.get("guardrail_flags", []),
        },
    )


def execute_rag_pipeline(query_input: QueryInput) -> Dict[str, Any]:
    """Runs the compiled LangGraph workflow with Labeled-Query Sidecar & Semantic Cache lookup."""
    t_start = time.perf_counter()

    # 0. Vetted Golden Labeled-Query Sidecar (Block / Instacart pattern)
    vetted = labeled_query_store.lookup(
        query_input.text, tenant_id=query_input.tenant_id, user_roles=query_input.user_roles
    )
    if vetted is not None:
        ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        trace = PipelineTrace(
            original_query=query_input.text,
            query_route="labeled_query_sidecar",
            retrieval_mode=query_input.retrieval_mode,
            tenant_id=query_input.tenant_id,
            user_roles=list(query_input.user_roles),
            cache_hit=True,
            total_latency_ms=ms,
        ).model_dump()
        out = {"answer": vetted["answer"], "tools": ["labeled_query_sidecar"], "trace": trace}
        _record_observability_and_audit(query_input, out)
        return out

    # 1. Role-Scoped Semantic Query Cache (skip when session_id has follow-up context)
    if query_input.enable_semantic_cache and not query_input.session_id:
        cached = semantic_cache.lookup(
            query=query_input.text,
            tenant_id=query_input.tenant_id,
            user_roles=query_input.user_roles,
            retrieval_mode=query_input.retrieval_mode,
            enable_web_search=query_input.enable_web_search,
        )
        if cached is not None:
            cache_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            trace = dict(cached["trace"])
            trace["cache_hit"] = True
            trace["total_latency_ms"] = cache_ms
            tools = list(cached["tools"])
            if "semantic_cache" not in tools:
                tools.insert(0, "semantic_cache")
            out = {"answer": cached["answer"], "tools": tools, "trace": trace}
            _record_observability_and_audit(query_input, out)
            return out

    final_state: RAGGraphState = rag_compiled_graph.invoke(_build_initial_state(query_input))
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    output = _package_output(query_input, final_state, total_ms)

    if query_input.enable_semantic_cache and not query_input.session_id:
        contributing_docs = {
            c.get("doc_id")
            for c in (final_state.get("relevant_chunks") or [])
            if c.get("doc_id")
        }
        semantic_cache.store(
            query=query_input.text,
            tenant_id=query_input.tenant_id,
            user_roles=query_input.user_roles,
            retrieval_mode=query_input.retrieval_mode,
            enable_web_search=query_input.enable_web_search,
            response_payload=output,
            doc_ids=contributing_docs,
        )

    _record_observability_and_audit(query_input, output)
    return output


def stream_rag_pipeline_sse(query_input: QueryInput) -> Iterator[str]:
    """Yields Server-Sent Events (SSE) for real-time LangGraph node progression and final output."""
    t_start = time.perf_counter()

    if query_input.enable_semantic_cache and not query_input.session_id:
        cached = semantic_cache.lookup(
            query=query_input.text,
            tenant_id=query_input.tenant_id,
            user_roles=query_input.user_roles,
            retrieval_mode=query_input.retrieval_mode,
            enable_web_search=query_input.enable_web_search,
        )
        if cached is not None:
            cache_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
            trace = dict(cached["trace"])
            trace["cache_hit"] = True
            trace["total_latency_ms"] = cache_ms
            out = {"answer": cached["answer"], "tools": ["semantic_cache"] + cached["tools"], "trace": trace}
            _record_observability_and_audit(query_input, out)
            yield f"data: {json.dumps({'event': 'stage', 'node': 'semantic_cache', 'status': 'hit', 'latency_ms': cache_ms})}\n\n"
            yield f"data: {json.dumps({'event': 'final', **out})}\n\n"
            return

    state = _build_initial_state(query_input)
    for step in rag_compiled_graph.stream(state):
        for node_name, updates in step.items():
            if isinstance(updates, dict):
                state.update(updates)
            yield f"data: {json.dumps({'event': 'stage', 'node': node_name, 'status': 'completed', 'query_route': state.get('query_route'), 'crag_decision': state.get('crag_decision')})}\n\n"

    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    output = _package_output(query_input, state, total_ms)

    if query_input.enable_semantic_cache and not query_input.session_id:
        contributing_docs = {
            c.get("doc_id")
            for c in (state.get("relevant_chunks") or [])
            if c.get("doc_id")
        }
        semantic_cache.store(
            query=query_input.text,
            tenant_id=query_input.tenant_id,
            user_roles=query_input.user_roles,
            retrieval_mode=query_input.retrieval_mode,
            enable_web_search=query_input.enable_web_search,
            response_payload=output,
            doc_ids=contributing_docs,
        )

    _record_observability_and_audit(query_input, output)
    yield f"data: {json.dumps({'event': 'final', **output})}\n\n"
