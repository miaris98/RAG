"""Production Observability:
1. CircuitBreaker (CLOSED -> OPEN -> HALF_OPEN) for external dependencies (Ollama LLM, DuckDuckGo).
2. Rolling Latency (p50, p95, p99) & RAG Telemetry Counters exposed at GET /metrics.
"""

import threading
import time
from collections import deque
from typing import Deque, Dict


class CircuitBreaker:
    """Thread-safe Circuit Breaker preventing cascading latency spikes when dependencies are down."""

    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout_sec: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self._lock = threading.Lock()
        self.state: str = "CLOSED"
        self.failure_count: int = 0
        self.opened_at: float = 0.0

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == "CLOSED":
                return True
            if self.state == "OPEN":
                if (time.time() - self.opened_at) >= self.recovery_timeout_sec:
                    self.state = "HALF_OPEN"
                    return True
                return False
            return True  # HALF_OPEN allows probe

    def record_success(self) -> None:
        with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold or self.state == "HALF_OPEN":
                self.state = "OPEN"
                self.opened_at = time.time()

    def status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
            }


def _percentile(sorted_vals: list, pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return round(float(sorted_vals[idx]), 2)


class MetricsCollector:
    """Collects rolling RAG latency percentiles and operational counters."""

    def __init__(self, window_size: int = 500):
        self._lock = threading.RLock()
        self._latencies_ms: Deque[float] = deque(maxlen=window_size)
        self.total_queries: int = 0
        self.cache_hits: int = 0
        self.crag_fallbacks: int = 0
        self.self_rag_retries: int = 0
        self.guardrail_triggers: int = 0
        self.gdpr_erasures: int = 0

    def record_query(
        self,
        latency_ms: float,
        cache_hit: bool = False,
        crag_decision: str = "sufficient",
        retry_count: int = 0,
        guardrail_flags: list = None,
    ) -> None:
        with self._lock:
            self.total_queries += 1
            self._latencies_ms.append(float(latency_ms))
            if cache_hit:
                self.cache_hits += 1
            if "fallback" in crag_decision:
                self.crag_fallbacks += 1
            self.self_rag_retries += int(retry_count)
            if guardrail_flags:
                self.guardrail_triggers += 1

    def record_gdpr_erasure(self) -> None:
        with self._lock:
            self.gdpr_erasures += 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            vals = sorted(self._latencies_ms)
            total = max(1, self.total_queries)
            return {
                "total_queries": self.total_queries,
                "cache_hit_rate": round(self.cache_hits / total, 4),
                "crag_fallback_rate": round(self.crag_fallbacks / total, 4),
                "self_rag_retries_total": self.self_rag_retries,
                "guardrail_triggers_total": self.guardrail_triggers,
                "gdpr_erasures_total": self.gdpr_erasures,
                "latency_ms": {
                    "p50": _percentile(vals, 50),
                    "p95": _percentile(vals, 95),
                    "p99": _percentile(vals, 99),
                    "sample_count": len(vals),
                },
                "circuit_breakers": {
                    "ollama_llm": llm_circuit_breaker.status(),
                    "duckduckgo_web": web_circuit_breaker.status(),
                },
            }

    def format_prometheus(
        self,
        corpus_stats: Dict[str, object],
        chain_valid: bool,
        trace_stats: Dict[str, object],
    ) -> str:
        """Renders operational, latency, corpus, audit-chain, and circuit-breaker metrics in Prometheus text format."""
        snap = self.snapshot()
        lat = snap["latency_ms"]  # type: ignore[index]
        cbs = snap["circuit_breakers"]  # type: ignore[index]
        llm_open = 1 if cbs["ollama_llm"]["state"] == "OPEN" else 0  # type: ignore[index]
        web_open = 1 if cbs["duckduckgo_web"]["state"] == "OPEN" else 0  # type: ignore[index]

        with self._lock:
            lines = [
                "# HELP rag_queries_total Total number of RAG queries executed.",
                "# TYPE rag_queries_total counter",
                f"rag_queries_total {self.total_queries}",
                "# HELP rag_cache_hits_total Total number of semantic/sidecar cache hits.",
                "# TYPE rag_cache_hits_total counter",
                f"rag_cache_hits_total {self.cache_hits}",
                "# HELP rag_crag_fallbacks_total Total number of CRAG external web fallback triggers.",
                "# TYPE rag_crag_fallbacks_total counter",
                f"rag_crag_fallbacks_total {self.crag_fallbacks}",
                "# HELP rag_self_rag_retries_total Total cyclic Self-RAG retry loops triggered.",
                "# TYPE rag_self_rag_retries_total counter",
                f"rag_self_rag_retries_total {self.self_rag_retries}",
                "# HELP rag_guardrail_triggers_total Total security guardrail triggers (PII / prompt injection).",
                "# TYPE rag_guardrail_triggers_total counter",
                f"rag_guardrail_triggers_total {self.guardrail_triggers}",
                "# HELP rag_gdpr_erasures_total Total GDPR Article 17 surgical document erasures.",
                "# TYPE rag_gdpr_erasures_total counter",
                f"rag_gdpr_erasures_total {self.gdpr_erasures}",
                "# HELP rag_latency_ms_p50 Rolling 50th percentile query latency in milliseconds.",
                "# TYPE rag_latency_ms_p50 gauge",
                f"rag_latency_ms_p50 {lat['p50']}",  # type: ignore[index]
                "# HELP rag_latency_ms_p95 Rolling 95th percentile query latency in milliseconds.",
                "# TYPE rag_latency_ms_p95 gauge",
                f"rag_latency_ms_p95 {lat['p95']}",  # type: ignore[index]
                "# HELP rag_latency_ms_p99 Rolling 99th percentile query latency in milliseconds.",
                "# TYPE rag_latency_ms_p99 gauge",
                f"rag_latency_ms_p99 {lat['p99']}",  # type: ignore[index]
                "# HELP rag_corpus_documents_total Active documents indexed in the RAG corpus.",
                "# TYPE rag_corpus_documents_total gauge",
                f"rag_corpus_documents_total {corpus_stats.get('total_documents', 0)}",
                "# HELP rag_corpus_chunks_total Active child chunks indexed across dense, BM25, and graph stores.",
                "# TYPE rag_corpus_chunks_total gauge",
                f"rag_corpus_chunks_total {corpus_stats.get('total_chunks', 0)}",
                "# HELP rag_traces_total Total distributed Langfuse-compatible traces stored.",
                "# TYPE rag_traces_total gauge",
                f"rag_traces_total {trace_stats.get('total_traces', 0)}",
                "# HELP rag_audit_chain_valid Cryptographic SHA-256 audit log chain integrity (1=valid, 0=tampered).",
                "# TYPE rag_audit_chain_valid gauge",
                f"rag_audit_chain_valid {1 if chain_valid else 0}",
                "# HELP rag_circuit_breaker_open Circuit breaker state per downstream service (1=OPEN, 0=CLOSED/HALF_OPEN).",
                "# TYPE rag_circuit_breaker_open gauge",
                f'rag_circuit_breaker_open{{service="ollama_llm"}} {llm_open}',
                f'rag_circuit_breaker_open{{service="duckduckgo_web"}} {web_open}',
            ]
        from app.observability.feedback import feedback_store

        fb = feedback_store.summary(limit=1)
        lines.extend(
            [
                "# HELP rag_user_feedback_total Total thumbs-up and thumbs-down user feedback ratings.",
                "# TYPE rag_user_feedback_total counter",
                f'rag_user_feedback_total{{rating="up"}} {fb["thumbs_up"]}',
                f'rag_user_feedback_total{{rating="down"}} {fb["thumbs_down"]}',
                "# HELP rag_user_satisfaction_rate Rolling user thumbs-up satisfaction ratio [0.0, 1.0].",
                "# TYPE rag_user_satisfaction_rate gauge",
                f'rag_user_satisfaction_rate {fb["satisfaction_rate"]}',
            ]
        )
        return "\n".join(lines) + "\n"


llm_circuit_breaker = CircuitBreaker(name="ollama_llm", failure_threshold=2, recovery_timeout_sec=30.0)
web_circuit_breaker = CircuitBreaker(name="duckduckgo_web", failure_threshold=3, recovery_timeout_sec=30.0)
metrics_collector = MetricsCollector()
