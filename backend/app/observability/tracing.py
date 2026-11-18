"""Langfuse-Compatible Distributed Trace & Span Collector for End-to-End RAG Observability.

Records hierarchical traces and per-node spans for every RAG query (and cache hit),
sanitized via SecurityGuardrails. When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
are configured, asynchronously forwards trace batches to Langfuse's /api/public/ingestion API.
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


class TraceCollector:
    """Thread-safe store of Langfuse-compatible distributed RAG traces and spans."""

    def __init__(self, max_traces: int = 500):
        self._lock = threading.RLock()
        self._traces: Deque[Dict[str, Any]] = deque(maxlen=max_traces)
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._langfuse_synced_count: int = 0

    def _forward_to_langfuse(self, trace_record: Dict[str, Any]) -> str:
        """Forwards trace + spans to Langfuse if credentials are present; non-blocking 0.5s timeout."""
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
            batch_events.append(
                {
                    "id": str(uuid.uuid4()),
                    "type": "span-create",
                    "timestamp": now_iso,
                    "body": {
                        "id": sp["span_id"],
                        "traceId": trace_record["trace_id"],
                        "name": sp["name"],
                        "metadata": {"duration_ms": sp["duration_ms"]},
                    },
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
        """Builds and stores a Langfuse-compatible trace from a completed RAG pipeline output."""
        trace_id = str(uuid.uuid4())
        trace_data = output.get("trace") or {}
        clean_query, _ = guardrails.sanitize_text(query_input.text, redact_pii=True, block_injections=True)
        clean_answer, _ = guardrails.sanitize_text(output.get("answer", "")[:400], redact_pii=True, block_injections=True)

        stage_latencies = trace_data.get("stage_latencies_ms") or {}
        spans: List[Dict[str, Any]] = []
        if trace_data.get("cache_hit"):
            spans.append(
                {
                    "span_id": str(uuid.uuid4())[:12],
                    "name": "semantic_cache_lookup",
                    "duration_ms": float(trace_data.get("total_latency_ms", 0.0)),
                    "status": "HIT",
                }
            )
        else:
            for stage_name, duration_ms in stage_latencies.items():
                spans.append(
                    {
                        "span_id": str(uuid.uuid4())[:12],
                        "name": stage_name.removesuffix("_ms"),
                        "duration_ms": float(duration_ms),
                        "status": "OK",
                    }
                )

        if not spans:
            spans.append(
                {
                    "span_id": str(uuid.uuid4())[:12],
                    "name": "rag_execution",
                    "duration_ms": float(trace_data.get("total_latency_ms", 0.0)),
                    "status": "OK",
                }
            )

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
            "total_latency_ms": float(trace_data.get("total_latency_ms", 0.0)),
            "spans": spans,
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
