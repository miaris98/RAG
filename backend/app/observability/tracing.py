"""Langfuse-Compatible Distributed Trace & Hierarchical Span Waterfall Collector.

Records hierarchical parent-child traces and nested retrieval sub-spans for every RAG query
(and cache hit), builds a proportional Gantt waterfall timeline, and optionally forwards batches
to Langfuse's /api/public/ingestion API.
"""

import base64
import os
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import requests

from app.guardrails.security import guardrails
from app.models import QueryInput


SUBSTAGE_SPAN_NAMES: Dict[str, str] = {
    "rbac_prefilter_ms": "retrieval.rbac_prefilter",
    "dense_search_ms": "retrieval.dense_vector_search",
    "bm25_search_ms": "retrieval.sparse_bm25_search",
    "entity_graph_ms": "retrieval.entity_graph_traversal",
    "rrf_and_parent_promotion_ms": "retrieval.rrf_and_parent_promotion",
    "cross_encoder_and_mmr_ms": "retrieval.cross_encoder_and_mmr",
}


def _render_waterfall_markdown(spans: List[Dict[str, Any]], total_ms: float) -> str:
    """Formats hierarchical spans into a proportional visual Gantt waterfall table."""
    safe_total = max(0.01, float(total_ms))
    lines = [
        "| Span Hierarchy | Offset | Duration | Timeline Waterfall | Key Attributes |",
        "| :--- | :---: | :---: | :--- | :--- |",
    ]
    for sp in spans:
        depth = int(sp.get("depth", 0))
        prefix = "" if depth == 0 else ("├─ " if depth == 1 else "│  └─ ")
        dur = float(sp.get("duration_ms", 0.0))
        bar_len = max(1, min(18, int(round((dur / safe_total) * 18))))
        bar = "█" * bar_len
        attrs = sp.get("attributes") or {}
        attr_str = ", ".join(f"`{k}={v}`" for k, v in list(attrs.items())[:3]) or "—"
        lines.append(
            f"| {prefix}`{sp['name']}` | `{sp.get('start_offset_ms', 0.0):.2f} ms` | "
            f"`{dur:.2f} ms` | `{bar}` | {attr_str} |"
        )
    return "\n".join(lines)


class TraceCollector:
    """Thread-safe store of Langfuse-compatible hierarchical RAG traces and spans."""

    def __init__(self, max_traces: int = 500):
        self._lock = threading.RLock()
        self._traces: Deque[Dict[str, Any]] = deque(maxlen=max_traces)
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._langfuse_synced_count: int = 0

    def _forward_to_langfuse(self, trace_record: Dict[str, Any]) -> str:
        """Forwards trace + hierarchical spans to Langfuse if credentials are present; non-blocking 0.5s timeout."""
        pub_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        sec_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        if not pub_key or not sec_key:
            return "local_trace_store"

        host = os.getenv("LANGFUSE_HOST", "http://localhost:3001").rstrip("/")
        token = base64.b64encode(f"{pub_key}:{sec_key}".encode("utf-8")).decode("ascii")
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        batch_events = [
            {
                "id": str(uuid.uuid4()),
                "type": "trace-create",
                "timestamp": now_iso,
                "body": {
                    "id": trace_record["trace_id"],
                    "name": trace_record["name"],
                    "userId": f"{trace_record['tenant_id']}:{','.join(trace_record['user_roles'])}",
                    "input": {"query": trace_record["query"]},
                    "output": {"answer_preview": trace_record["answer_preview"]},
                    "metadata": {
                        "query_route": trace_record["query_route"],
                        "retrieval_mode": trace_record["retrieval_mode"],
                        "cache_hit": trace_record["cache_hit"],
                        "crag_decision": trace_record["crag_decision"],
                        "faithfulness_score": trace_record["faithfulness_score"],
                        "total_latency_ms": trace_record["total_latency_ms"],
                    },
                    "tags": [trace_record["retrieval_mode"], trace_record["query_route"]],
                },
            }
        ]

        for sp in trace_record["spans"]:
            body: Dict[str, Any] = {
                "id": sp["span_id"],
                "traceId": trace_record["trace_id"],
                "name": sp["name"],
                "metadata": {
                    "duration_ms": sp["duration_ms"],
                    "start_offset_ms": sp.get("start_offset_ms", 0.0),
                    "attributes": sp.get("attributes", {}),
                },
            }
            if sp.get("parent_span_id"):
                body["parentObservationId"] = sp["parent_span_id"]
            batch_events.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "span-create",
                    "timestamp": now_iso,
                    "body": body,
                }
            )

        try:
            resp = requests.post(
                f"{host}/api/public/ingestion",
                json={"batch": batch_events},
                headers={"Authorization": f"Basic {token}"},
                timeout=0.5,
            )
            if resp.status_code < 300:
                with self._lock:
                    self._langfuse_synced_count += 1
                return "synced_to_langfuse"
            return f"langfuse_http_{resp.status_code}"
        except Exception:
            return "local_trace_store_langfuse_unreachable"

    def record_trace(self, query_input: QueryInput, output: Dict[str, Any]) -> str:
        """Builds and stores a hierarchical Langfuse-compatible trace and attaches waterfall_markdown."""
        trace_id = str(uuid.uuid4())
        trace_data = output.get("trace") or {}
        clean_query, _ = guardrails.sanitize_text(query_input.text, redact_pii=True, block_injections=True)
        clean_answer, _ = guardrails.sanitize_text(output.get("answer", "")[:400], redact_pii=True, block_injections=True)

        total_ms = float(trace_data.get("total_latency_ms", 0.0))
        stage_latencies = trace_data.get("stage_latencies_ms") or {}
        funnel = trace_data.get("retrieval_funnel") or {}
        substage_latencies = funnel.get("substage_latencies_ms") or {}

        root_span_id = str(uuid.uuid4())[:12]
        spans: List[Dict[str, Any]] = [
            {
                "span_id": root_span_id,
                "parent_span_id": None,
                "depth": 0,
                "name": "rag_query_pipeline",
                "start_offset_ms": 0.0,
                "duration_ms": round(total_ms, 2),
                "status": "OK",
                "attributes": {
                    "route": trace_data.get("query_route", "standard"),
                    "mode": query_input.retrieval_mode,
                    "cache_hit": bool(trace_data.get("cache_hit", False)),
                },
            }
        ]

        offset = 0.0
        if trace_data.get("cache_hit"):
            spans.append(
                {
                    "span_id": str(uuid.uuid4())[:12],
                    "parent_span_id": root_span_id,
                    "depth": 1,
                    "name": "semantic_cache_lookup",
                    "start_offset_ms": 0.0,
                    "duration_ms": round(total_ms, 2),
                    "status": "HIT",
                    "attributes": {"tenant_id": query_input.tenant_id},
                }
            )
        else:
            for stage_name, duration_ms in stage_latencies.items():
                clean_name = stage_name[:-3] if stage_name.endswith("_ms") else stage_name
                dur = float(duration_ms)
                stage_span_id = str(uuid.uuid4())[:12]
                stage_attrs: Dict[str, Any] = {}
                if clean_name == "query_routing":
                    stage_attrs = {"route": trace_data.get("query_route", "standard")}
                elif clean_name == "hybrid_retrieval_rerank":
                    stage_attrs = {
                        "authorized": funnel.get("rbac_authorized_chunks", 0),
                        "selected": funnel.get("mmr_selected_count", 0),
                    }
                elif clean_name == "crag_grading":
                    stage_attrs = {"decision": trace_data.get("crag_decision", "sufficient")}
                elif clean_name == "self_rag_verification":
                    stage_attrs = {
                        "verdict": trace_data.get("self_rag_verdict", "grounded"),
                        "faithfulness": trace_data.get("faithfulness_score", 1.0),
                    }

                spans.append(
                    {
                        "span_id": stage_span_id,
                        "parent_span_id": root_span_id,
                        "depth": 1,
                        "name": clean_name,
                        "start_offset_ms": round(offset, 2),
                        "duration_ms": round(dur, 2),
                        "status": "OK",
                        "attributes": stage_attrs,
                    }
                )

                # Attach nested retrieval sub-spans under hybrid_retrieval_rerank
                if clean_name == "hybrid_retrieval_rerank" and substage_latencies:
                    sub_offset = offset
                    for sub_key, sub_name in SUBSTAGE_SPAN_NAMES.items():
                        sub_dur = float(substage_latencies.get(sub_key, 0.0))
                        sub_attrs: Dict[str, Any] = {}
                        if sub_key == "rbac_prefilter_ms":
                            sub_attrs = {"authorized": funnel.get("rbac_authorized_chunks", 0)}
                        elif sub_key == "dense_search_ms":
                            sub_attrs = {"candidates": funnel.get("dense_candidates_count", 0)}
                        elif sub_key == "bm25_search_ms":
                            sub_attrs = {"candidates": funnel.get("sparse_candidates_count", 0)}
                        elif sub_key == "entity_graph_ms":
                            sub_attrs = {"candidates": funnel.get("graph_candidates_count", 0)}
                        elif sub_key == "rrf_and_parent_promotion_ms":
                            sub_attrs = {
                                "rrf_pool": funnel.get("rrf_fused_pool_count", 0),
                                "parents": funnel.get("parent_promoted_count", 0),
                            }
                        elif sub_key == "cross_encoder_and_mmr_ms":
                            sub_attrs = {
                                "relevant": funnel.get("crag_relevant_count", 0),
                                "final_k": funnel.get("mmr_selected_count", 0),
                            }
                        spans.append(
                            {
                                "span_id": str(uuid.uuid4())[:12],
                                "parent_span_id": stage_span_id,
                                "depth": 2,
                                "name": sub_name,
                                "start_offset_ms": round(sub_offset, 2),
                                "duration_ms": round(sub_dur, 2),
                                "status": "OK",
                                "attributes": sub_attrs,
                            }
                        )
                        sub_offset += sub_dur

                offset += dur

        waterfall_md = _render_waterfall_markdown(spans, total_ms)
        trace_data["trace_id"] = trace_id
        trace_data["waterfall_markdown"] = waterfall_md

        retrieved_chunks = trace_data.get("retrieved_chunks") or []
        record: Dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": round(time.time(), 3),
            "name": "rag_query_pipeline",
            "query": clean_query,
            "answer_preview": clean_answer,
            "tenant_id": query_input.tenant_id,
            "user_roles": list(query_input.user_roles),
            "retrieval_mode": query_input.retrieval_mode,
            "query_route": trace_data.get("query_route", "standard"),
            "cache_hit": bool(trace_data.get("cache_hit", False)),
            "crag_decision": trace_data.get("crag_decision", "sufficient"),
            "self_rag_verdict": trace_data.get("self_rag_verdict", "grounded"),
            "faithfulness_score": float(trace_data.get("faithfulness_score", 1.0)),
            "citation_count": int(trace_data.get("citation_count", 0)),
            "guardrail_flags": list(trace_data.get("guardrail_flags") or []),
            "retrieved_chunk_ids": [c.get("chunk_id") for c in retrieved_chunks if c.get("chunk_id")],
            "retrieved_doc_ids": sorted({c.get("doc_id") for c in retrieved_chunks if c.get("doc_id")}),
            "retrieval_funnel": funnel,
            "citations": trace_data.get("citations") or [],
            "total_latency_ms": total_ms,
            "spans": spans,
            "waterfall_markdown": waterfall_md,
            "langfuse_status": "local_trace_store",
        }

        record["langfuse_status"] = self._forward_to_langfuse(record)

        with self._lock:
            if len(self._traces) >= (self._traces.maxlen or 500):
                oldest = self._traces.popleft()
                self._by_id.pop(oldest["trace_id"], None)
            self._traces.append(record)
            self._by_id[trace_id] = record

        return trace_id

    def record_multi_agent_trace(
        self,
        query: str,
        answer_preview: str,
        tenant_id: str,
        user_roles: List[str],
        retrieval_mode: str,
        subagent_spans: List[Dict[str, Any]],
        faithfulness_score: float,
        citation_count: int,
        guardrail_flags: List[str],
        total_latency_ms: float,
    ) -> tuple[str, str]:
        """Records a hierarchical Supervisor + Specialist Subagents distributed trace."""
        trace_id = f"tr_{uuid.uuid4().hex[:16]}"
        root_span_id = str(uuid.uuid4())[:12]
        clean_query, _ = guardrails.sanitize_text(query)
        clean_answer, _ = guardrails.sanitize_text(answer_preview[:280])

        spans: List[Dict[str, Any]] = [
            {
                "span_id": root_span_id,
                "parent_span_id": None,
                "depth": 0,
                "name": "supervisor_orchestrator",
                "start_offset_ms": 0.0,
                "duration_ms": round(total_latency_ms, 2),
                "status": "OK",
                "attributes": {
                    "subagents": len(subagent_spans),
                    "roles": ",".join(user_roles),
                },
            }
        ]
        offset = 0.0
        for sp in subagent_spans:
            dur = float(sp.get("duration_ms", 0.0))
            spans.append(
                {
                    "span_id": str(uuid.uuid4())[:12],
                    "parent_span_id": root_span_id,
                    "depth": 1,
                    "name": sp["name"],
                    "start_offset_ms": round(offset, 2),
                    "duration_ms": round(dur, 2),
                    "status": "OK",
                    "attributes": dict(sp.get("attributes") or {}),
                }
            )
            offset += dur

        waterfall_md = _render_waterfall_markdown(spans, total_latency_ms)
        record: Dict[str, Any] = {
            "trace_id": trace_id,
            "timestamp": round(time.time(), 3),
            "name": "supervisor_orchestrator",
            "query": clean_query,
            "answer_preview": clean_answer,
            "tenant_id": tenant_id,
            "user_roles": list(user_roles),
            "retrieval_mode": retrieval_mode,
            "query_route": "multi_agent_supervisor",
            "cache_hit": False,
            "crag_decision": "multi_agent",
            "self_rag_verdict": "audited",
            "faithfulness_score": float(faithfulness_score),
            "citation_count": int(citation_count),
            "guardrail_flags": list(guardrail_flags),
            "retrieved_chunk_ids": [],
            "retrieved_doc_ids": [],
            "retrieval_funnel": {},
            "citations": [],
            "total_latency_ms": round(total_latency_ms, 2),
            "spans": spans,
            "waterfall_markdown": waterfall_md,
            "langfuse_status": "local_trace_store",
        }
        record["langfuse_status"] = self._forward_to_langfuse(record)

        with self._lock:
            if len(self._traces) >= (self._traces.maxlen or 500):
                oldest = self._traces.popleft()
                self._by_id.pop(oldest["trace_id"], None)
            self._traces.append(record)
            self._by_id[trace_id] = record

        return trace_id, waterfall_md

    def list_traces(self, limit: int = 50) -> Dict[str, Any]:
        with self._lock:
            recent = list(reversed(list(self._traces)[-limit:]))
            return {
                "total_traces": len(self._traces),
                "langfuse_configured": bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")),
                "langfuse_synced_count": self._langfuse_synced_count,
                "traces": recent,
            }

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._by_id.get(trace_id)
            return dict(rec) if rec is not None else None

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_traces": len(self._traces),
                "langfuse_synced_count": self._langfuse_synced_count,
            }

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()
            self._by_id.clear()
            self._langfuse_synced_count = 0


trace_collector = TraceCollector()
