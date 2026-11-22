"""TDD tests for P4.1: Retries with Exponential Backoff & Jitter, Bulkhead Isolation, and Graceful Degradation Matrix."""

import threading
import time
from fastapi.testclient import TestClient

from main import app
from app.observability.metrics import llm_circuit_breaker
from app.resilience.patterns import (
    BulkheadPool,
    BulkheadRejectedError,
    bulkhead_registry,
    evaluate_degradation_level,
    retry_with_backoff,
)


client = TestClient(app)


def test_retry_with_exponential_backoff_and_jitter_recovers_transient_faults():
    """Verifies exponential backoff + jitter delays grow across transient attempts and succeed upon recovery."""
    attempts = {"count": 0}
    recorded_delays: list[float] = []

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ConnectionError("transient network drop")
        return "recovered_ok"

    result = retry_with_backoff(
        flaky_call,
        max_attempts=4,
        base_delay_sec=0.05,
        max_delay_sec=1.0,
        jitter_ratio=0.25,
        sleep_fn=lambda d: recorded_delays.append(d),
    )
    assert result == "recovered_ok"
    assert attempts["count"] == 3
    assert len(recorded_delays) == 2
    assert recorded_delays[0] >= 0.05
    assert recorded_delays[1] >= 0.10


def test_bulkhead_isolates_dependency_concurrency_and_protects_other_pools():
    """Saturating Service A's bulkhead rejects excess calls without affecting Service B's bulkhead."""
    pool_a = BulkheadPool(name="service_a", max_concurrent=1)
    pool_b = BulkheadPool(name="service_b", max_concurrent=2)

    entered = threading.Event()
    release = threading.Event()

    def slow_service_a():
        with pool_a.slot(timeout_sec=0.05):
            entered.set()
            release.wait(timeout=2.0)

    t = threading.Thread(target=slow_service_a)
    t.start()
    assert entered.wait(timeout=1.0)

    try:
        # Excess call to saturated Service A must be rejected immediately by Bulkhead
        rejected = False
        try:
            with pool_a.slot(timeout_sec=0.02):
                pass
        except BulkheadRejectedError:
            rejected = True
        assert rejected is True
        assert pool_a.status()["total_rejected"] == 1

        # Service B remains completely unaffected (isolated compartment)
        with pool_b.slot(timeout_sec=0.05):
            assert pool_b.status()["in_flight"] == 1
    finally:
        release.set()
        t.join(timeout=2.0)


def test_graceful_degradation_endpoint_and_prometheus_bulkhead_metrics():
    """GET /health/resilience reports degradation tier, bulkheads, and breakers; Prometheus exports bulkhead gauges."""
    try:
        llm_circuit_breaker.record_failure()
        llm_circuit_breaker.record_failure()
        deg = evaluate_degradation_level()
        assert deg["level"] in ("L1_EXTRACTIVE_DEGRADED", "L2_LOCAL_HYBRID_FALLBACK")
        assert deg["core_rag_available"] is True

        resp = client.get("/health/resilience")
        assert resp.status_code == 200
        data = resp.json()
        assert "degradation" in data
        assert "bulkheads" in data
        assert "ollama_llm" in data["bulkheads"]
        assert "milvus_vector" in data["bulkheads"]

        prom = client.get("/metrics/prometheus")
        assert prom.status_code == 200
        assert "rag_bulkhead_saturation_ratio" in prom.text
        assert "rag_bulkhead_rejected_total" in prom.text
    finally:
        llm_circuit_breaker.record_success()
