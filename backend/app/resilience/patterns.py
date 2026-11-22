"""P4.1 Resilience & Fault Tolerance: Exponential Backoff + Jitter, Bulkhead Pools, and Graceful Degradation."""

from __future__ import annotations

from contextlib import contextmanager
import random
import threading
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type, TypeVar

from app.observability.metrics import CircuitBreaker, llm_circuit_breaker, web_circuit_breaker


T = TypeVar("T")


class BulkheadRejectedError(RuntimeError):
    """Raised when a downstream dependency's concurrency bulkhead is saturated."""


class BulkheadPool:
    """Thread-safe concurrency compartment isolating one downstream dependency from others."""

    def __init__(self, name: str, max_concurrent: int = 8):
        self.name = name
        self.max_concurrent = max(1, int(max_concurrent))
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._total_acquired = 0
        self._total_rejected = 0

    @contextmanager
    def slot(self, timeout_sec: float = 0.05) -> Iterator[None]:
        acquired = self._sem.acquire(timeout=max(0.0, float(timeout_sec)))
        if not acquired:
            with self._lock:
                self._total_rejected += 1
            raise BulkheadRejectedError(
                f"Bulkhead '{self.name}' saturated (max_concurrent={self.max_concurrent})"
            )
        with self._lock:
            self._in_flight += 1
            self._total_acquired += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_flight = max(0, self._in_flight - 1)
            self._sem.release()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            sat = round(self._in_flight / self.max_concurrent, 4)
            return {
                "name": self.name,
                "max_concurrent": self.max_concurrent,
                "in_flight": self._in_flight,
                "saturation_ratio": sat,
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
            }


class BulkheadRegistry:
    """Registry of isolated dependency bulkheads across the RAG microservices gateway."""

    def __init__(self):
        self._pools: Dict[str, BulkheadPool] = {
            "milvus_vector": BulkheadPool("milvus_vector", max_concurrent=8),
            "ollama_llm": BulkheadPool("ollama_llm", max_concurrent=4),
            "sql_opsbot": BulkheadPool("sql_opsbot", max_concurrent=8),
            "web_search": BulkheadPool("web_search", max_concurrent=4),
        }

    def get(self, name: str) -> BulkheadPool:
        if name not in self._pools:
            self._pools[name] = BulkheadPool(name, max_concurrent=4)
        return self._pools[name]

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {name: pool.status() for name, pool in self._pools.items()}


class RetryTelemetry:
    """Tracks exponential backoff retries and recoveries across dependencies."""

    def __init__(self):
        self._lock = threading.Lock()
        self.total_retries = 0
        self.total_recoveries = 0
        self.total_exhausted = 0

    def record(self, retries_used: int, succeeded: bool) -> None:
        with self._lock:
            self.total_retries += retries_used
            if succeeded and retries_used > 0:
                self.total_recoveries += 1
            elif not succeeded:
                self.total_exhausted += 1

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total_retries": self.total_retries,
                "total_recoveries": self.total_recoveries,
                "total_exhausted": self.total_exhausted,
            }


bulkhead_registry = BulkheadRegistry()
retry_telemetry = RetryTelemetry()


def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 3,
    base_delay_sec: float = 0.05,
    max_delay_sec: float = 2.0,
    jitter_ratio: float = 0.3,
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    circuit_breaker: Optional[CircuitBreaker] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Executes fn with exponential backoff and randomized jitter, cooperating with an optional CircuitBreaker."""
    attempts = max(1, int(max_attempts))
    last_exc: Optional[BaseException] = None

    for attempt in range(attempts):
        if circuit_breaker is not None and not circuit_breaker.allow_request():
            raise RuntimeError(f"CircuitBreaker '{circuit_breaker.name}' is OPEN; skipping retry attempt.")
        try:
            res = fn()
            if circuit_breaker is not None:
                circuit_breaker.record_success()
            retry_telemetry.record(retries_used=attempt, succeeded=True)
            return res
        except retryable_exceptions as exc:
            last_exc = exc
            if circuit_breaker is not None:
                circuit_breaker.record_failure()
            if attempt + 1 >= attempts:
                retry_telemetry.record(retries_used=attempt, succeeded=False)
                raise
            exp_delay = min(max_delay_sec, base_delay_sec * (2 ** attempt))
            jitter = random.uniform(0.0, exp_delay * max(0.0, jitter_ratio))
            sleep_fn(exp_delay + jitter)

    assert last_exc is not None
    raise last_exc


def evaluate_degradation_level() -> Dict[str, Any]:
    """Computes live Graceful Degradation Tier (L0-L3) from breakers, vector store, and bulkheads."""
    from app.retrieval.hybrid import retrieval_engine

    llm_open = llm_circuit_breaker.status()["state"] == "OPEN"
    web_open = web_circuit_breaker.status()["state"] == "OPEN"
    milvus_up = bool(retrieval_engine._milvus_connected)
    bh_snap = bulkhead_registry.snapshot()
    saturated = [k for k, v in bh_snap.items() if v["saturation_ratio"] >= 1.0]

    if len(saturated) >= 2:
        level = "L3_CACHE_ONLY_SURVIVAL"
        summary = "Multiple downstream bulkheads saturated; serving Semantic & Labeled-Query caches."
    elif llm_open or web_open:
        level = "L1_EXTRACTIVE_DEGRADED"
        summary = "LLM or Web breaker OPEN; gracefully degraded to citation-grounded extractive synthesis."
    elif not milvus_up:
        level = "L2_LOCAL_HYBRID_FALLBACK"
        summary = "External Milvus offline; serving full Dense + BM25 + Entity Graph from local hybrid tier."
    else:
        level = "L0_FULL_OPERATIONAL"
        summary = "All dependencies healthy; full Tri-Modal GraphRAG + LLM + Web Search active."

    return {
        "level": level,
        "summary": summary,
        "core_rag_available": True,
        "milvus_connected": milvus_up,
        "llm_breaker_state": llm_circuit_breaker.status()["state"],
        "web_breaker_state": web_circuit_breaker.status()["state"],
        "saturated_bulkheads": saturated,
    }
