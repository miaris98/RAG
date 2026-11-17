"""Compiles and executes the LangGraph Corrective RAG (CRAG) + Self-RAG StateGraph."""

import time
from typing import Any, Callable, Dict

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
            while curr and curr != END:
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
    synthesize_cited_answer,
    verify_faithfulness_self_rag,
)
from app.agent.state import RAGGraphState
from app.models import PipelineTrace, QueryInput, RetrievedChunk


def build_rag_state_graph():
    """Constructs the explicit LangGraph StateGraph for CRAG + Self-RAG orchestration."""
    workflow = StateGraph(RAGGraphState)

    # Register explicit pipeline nodes
    workflow.add_node("analyze_and_expand_query", analyze_and_expand_query)
    workflow.add_node("retrieve_and_rerank", retrieve_and_rerank)
    workflow.add_node("grade_retrieval_crag", grade_retrieval_crag)
    workflow.add_node("external_web_search", external_web_search)
    workflow.add_node("synthesize_cited_answer", synthesize_cited_answer)
    workflow.add_node("verify_faithfulness_self_rag", verify_faithfulness_self_rag)

    # Wire deterministic and conditional edges
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
    workflow.add_edge("verify_faithfulness_self_rag", END)

    return workflow.compile()


rag_compiled_graph = build_rag_state_graph()


def execute_rag_pipeline(query_input: QueryInput) -> Dict[str, Any]:
    """Runs the compiled LangGraph workflow and packages the answer, tool audit, and PipelineTrace."""
    t_start = time.perf_counter()
    initial_state: RAGGraphState = {
        "query": query_input.text,
        "retrieval_mode": query_input.retrieval_mode,
        "top_k": query_input.top_k,
        "enable_query_expansion": query_input.enable_query_expansion,
        "enable_web_search": query_input.enable_web_search,
        "enable_self_rag": query_input.enable_self_rag,
        "tools_used": [],
        "stage_latencies_ms": {},
    }

    final_state: RAGGraphState = rag_compiled_graph.invoke(initial_state)
    total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    retrieved_raw = final_state.get("retrieved_chunks") or []
    relevant_raw = final_state.get("relevant_chunks") or []
    retrieved_models = [RetrievedChunk(**c) for c in retrieved_raw]

    trace = PipelineTrace(
        original_query=query_input.text,
        expanded_queries=final_state.get("expanded_queries") or [query_input.text],
        retrieval_mode=query_input.retrieval_mode,
        candidates_retrieved_count=len(retrieved_models),
        relevant_chunks_count=len(relevant_raw),
        crag_decision=final_state.get("crag_decision", "sufficient"),
        self_rag_verdict=final_state.get("self_rag_verdict", "grounded"),
        faithfulness_score=float(final_state.get("faithfulness_score", 1.0)),
        citation_count=int(final_state.get("citation_count", 0)),
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
