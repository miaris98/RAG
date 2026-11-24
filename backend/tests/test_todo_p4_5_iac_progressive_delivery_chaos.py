"""TDD tests for P4.5: Immutable IaC, Progressive Delivery (Canary/Blue-Green), and Chaos Engineering."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from main import app
from app.observability.metrics import llm_circuit_breaker


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_immutable_iac_kubernetes_and_terraform_manifests_valid():
    k8s_path = REPO_ROOT / "deploy" / "k8s" / "rag-deployment-hpa-netpol.yaml"
    tf_path = REPO_ROOT / "deploy" / "terraform" / "main.tf"

    assert k8s_path.exists(), f"Missing Kubernetes IaC manifest: {k8s_path}"
    assert tf_path.exists(), f"Missing Terraform IaC manifest: {tf_path}"

    k8s_text = k8s_path.read_text(encoding="utf-8")
    for required_token in (
        "kind: Deployment",
        "kind: Service",
        "kind: HorizontalPodAutoscaler",
        "kind: NetworkPolicy",
        "kind: PodDisruptionBudget",
        "readOnlyRootFilesystem: true",
        "runAsNonRoot: true",
    ):
        assert required_token in k8s_text, f"K8s manifest missing '{required_token}'"

    tf_text = tf_path.read_text(encoding="utf-8")
    for required_token in ("terraform {", "variable ", "resource ", "rag_gateway"):
        assert required_token in tf_text, f"Terraform manifest missing '{required_token}'"


def test_progressive_delivery_canary_feature_flags_and_blue_green_header():
    from app.resilience.progressive_delivery import FeatureFlagManager, feature_flags

    mgr = FeatureFlagManager()
    mgr.set_flag("hyde_v2_canary", enabled=True, rollout_percentage=50)

    # Deterministic bucketing: same subject always gets same canary decision
    d1 = mgr.evaluate("hyde_v2_canary", subject_key="tenant-alpha")
    d2 = mgr.evaluate("hyde_v2_canary", subject_key="tenant-alpha")
    assert d1 == d2

    # 50% rollout across 200 distinct subjects should bucket roughly near half (25%..75%)
    hits = sum(1 for i in range(200) if mgr.evaluate("hyde_v2_canary", subject_key=f"user-{i}"))
    assert 50 <= hits <= 150, f"Unexpected 50% canary distribution: {hits}/200"

    mgr.set_flag("hyde_v2_canary", enabled=True, rollout_percentage=100)
    assert mgr.evaluate("hyde_v2_canary", subject_key="any-user") is True
    mgr.set_flag("hyde_v2_canary", enabled=False, rollout_percentage=100)
    assert mgr.evaluate("hyde_v2_canary", subject_key="any-user") is False

    # HTTP endpoints + Blue-Green X-Release-Track header via APIGatewayMiddleware
    original_track = feature_flags.active_release_track
    client = TestClient(app)
    llm_circuit_breaker.record_success()
    try:
        res_set = client.post(
            "/ops/feature-flags",
            json={
                "flag_name": "rag_canary_reranker",
                "enabled": True,
                "rollout_percentage": 100,
                "active_release_track": "green",
            },
        )
        assert res_set.status_code == 200, res_set.text
        body = res_set.json()
        assert body["active_release_track"] == "green"
        assert body["flags"]["rag_canary_reranker"]["rollout_percentage"] == 100

        res_get = client.get("/ops/feature-flags", params={"subject_key": "demo-user"})
        assert res_get.status_code == 200
        assert res_get.headers.get("X-Release-Track") == "green"
        assert res_get.json()["evaluations"]["rag_canary_reranker"] is True
    finally:
        feature_flags.set_release_track(original_track)


def test_chaos_engineering_fault_injector_triggers_resilience_and_resets_cleanly():
    from app.resilience.chaos import ChaosInjectedError, chaos_controller
    from app.resilience.patterns import retry_with_backoff

    client = TestClient(app)
    llm_circuit_breaker.record_success()
    try:
        # Enable controlled 100% failure chaos on ollama target
        r_enable = client.post(
            "/ops/chaos",
            json={
                "enabled": True,
                "target": "ollama",
                "mode": "failure",
                "failure_rate": 1.0,
                "latency_ms": 0,
            },
        )
        assert r_enable.status_code == 200, r_enable.text
        assert r_enable.json()["enabled"] is True

        # Drive failures through retry_with_backoff + llm_circuit_breaker -> trips breaker OPEN and short-circuits
        def _call_ollama():
            chaos_controller.maybe_inject("ollama")
            return "ok"

        with pytest.raises((ChaosInjectedError, RuntimeError)):
            retry_with_backoff(
                _call_ollama,
                max_attempts=llm_circuit_breaker.failure_threshold + 1,
                base_delay_sec=0.001,
                jitter_ratio=0.0,
                circuit_breaker=llm_circuit_breaker,
                sleep_fn=lambda _: None,
            )

        assert chaos_controller.snapshot()["total_faults_injected"] >= llm_circuit_breaker.failure_threshold
        assert llm_circuit_breaker.status()["state"] == "OPEN"

        res_health = client.get("/health/resilience")
        assert res_health.status_code == 200
        assert res_health.json()["degradation"]["level"] == "L1_EXTRACTIVE_DEGRADED"

        # Latency mode injects bounded delay without raising
        sleeps = []
        r_lat = client.post(
            "/ops/chaos",
            json={
                "enabled": True,
                "target": "milvus",
                "mode": "latency",
                "failure_rate": 0.0,
                "latency_ms": 25,
            },
        )
        assert r_lat.status_code == 200
        chaos_controller.maybe_inject("milvus", sleep_fn=lambda s: sleeps.append(s))
        assert len(sleeps) == 1 and abs(sleeps[0] - 0.025) < 1e-6

        # Disable chaos cleanly
        r_off = client.post(
            "/ops/chaos",
            json={"enabled": False, "target": "ollama", "mode": "failure", "failure_rate": 0.0, "latency_ms": 0},
        )
        assert r_off.status_code == 200
        assert client.get("/ops/chaos").json()["enabled"] is False
    finally:
        chaos_controller.reset()
        llm_circuit_breaker.record_success()


def test_all_p0_through_p4_roadmap_todos_100_percent_complete():
    todo_doc = (REPO_ROOT / "docs" / "INDUSTRY_RAG_TODO.md").read_text(encoding="utf-8")
    assert "- [ ]" not in todo_doc, "Found remaining unchecked '- [ ]' item in docs/INDUSTRY_RAG_TODO.md"
