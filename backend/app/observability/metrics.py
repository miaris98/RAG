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


llm_circuit_breaker = CircuitBreaker(name="ollama_llm", failure_threshold=2, recovery_timeout_sec=30.0)
web_circuit_breaker = CircuitBreaker(name="duckduckgo_web", failure_threshold=3, recovery_timeout_sec=30.0)
metrics_collector = MetricsCollector()
