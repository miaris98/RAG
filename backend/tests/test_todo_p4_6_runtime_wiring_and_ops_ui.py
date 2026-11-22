"""TDD tests for P4.6: End-to-End Microservices Runtime Wiring & Gradio Ops UI Parity."""

from pathlib import Path
from fastapi.testclient import TestClient

from main import app
from app.guardrails.audit import audit_logger
from app.observability.metrics import llm_circuit_breaker
from app.resilience import bulkhead_registry, chaos_controller
from app.security import secrets_provider


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_p4_6_live_rag_pipeline_uses_bulkheads_retries_and_chaos_injection():
    client = TestClient(app)
    llm_circuit_breaker.record_success()
    chaos_controller.reset()

    try:
        r_seed = client.post("/ingest-sample")
        assert r_seed.status_code == 200, r_seed.text

        before_milvus = bulkhead_registry.get("milvus_vector").status()["total_acquired"]
        before_ollama = bulkhead_registry.get("ollama_llm").status()["total_acquired"]
        before_faults = chaos_controller.snapshot()["total_faults_injected"]

        # Enable live chaos fault on Ollama so /query-sync exercises chaos + bulkhead + graceful fallback
        r_chaos = client.post(
            "/ops/chaos",
            json={
                "enabled": True,
                "target": "ollama",
                "mode": "failure",
                "failure_rate": 1.0,
                "latency_ms": 0,
            },
        )
        assert r_chaos.status_code == 200

        r_query = client.post(
            "/query-stream",
            json={
                "text": "What triggers ERR_LEDGER_9042?",
                "retrieval_mode": "hybrid_graph_rerank",
                "top_k": 3,
                "tenant_id": "default",
                "user_roles": ["public", "engineering"],
                "enable_web_search": False,
                "enable_semantic_cache": False,
            },
        )
        assert r_query.status_code == 200, r_query.text
        assert '"event": "final"' in r_query.text, "Expected SSE final grounded fallback event under Ollama chaos"

        after_milvus = bulkhead_registry.get("milvus_vector").status()["total_acquired"]
        after_ollama = bulkhead_registry.get("ollama_llm").status()["total_acquired"]
        after_faults = chaos_controller.snapshot()["total_faults_injected"]

        assert after_milvus > before_milvus, "Dense/Milvus retrieval did not acquire milvus_vector bulkhead"
        assert after_ollama > before_ollama, "LLM synthesis did not acquire ollama_llm bulkhead"
        assert after_faults > before_faults, "Ollama chaos fault was not invoked during live /query-sync"
    finally:
        chaos_controller.reset()
        llm_circuit_breaker.record_success()


def test_p4_6_a2a_endpoint_enforces_zero_trust_service_token_when_present():
    client = TestClient(app)
    r_seed = client.post("/ingest-sample")
    assert r_seed.status_code == 200, r_seed.text

    # 1. Issue a valid Zero Trust service JWT
    tok_resp = client.post(
        "/security/token",
        json={
            "subject": "peer-ops-agent",
            "audience": "rag-gateway-api",
            "roles": ["public", "engineering"],
            "ttl_sec": 120.0,
        },
    )
    assert tok_resp.status_code == 200
    token = tok_resp.json()["token"]

    rpc_payload = {
        "jsonrpc": "2.0",
        "id": "zt-a2a-1",
        "method": "tasks/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "What triggers ERR_LEDGER_9042?"}],
            },
            "metadata": {
                "tenant_id": "default",
                "user_roles": ["public", "engineering"],
                "enable_web_search": False,
            },
        },
    }

    # 2. First call with valid X-Service-Token succeeds
    r_ok = client.post(
        "/a2a",
        json=rpc_payload,
        headers={"X-Peer-Agent-Id": "external-ops-agent", "X-Service-Token": f"Bearer {token}"},
    )
    assert r_ok.status_code == 200, r_ok.text
    assert r_ok.json().get("result", {}).get("status", {}).get("state") == "completed"

    # 3. Replaying the exact same JWT on /a2a must be rejected (401 replay protection)
    r_replay = client.post(
        "/a2a",
        json=rpc_payload,
        headers={"X-Peer-Agent-Id": "external-ops-agent", "X-Service-Token": token},
    )
    assert r_replay.status_code == 401, f"Expected 401 on replayed JWT, got {r_replay.status_code}: {r_replay.text}"

    # 4. Invalid / forged JWT must be rejected (401)
    r_bad = client.post(
        "/a2a",
        json=rpc_payload,
        headers={"X-Service-Token": "invalid.forged.token"},
    )
    assert r_bad.status_code == 401


def test_p4_6_audit_logger_automatically_redacts_managed_secrets():
    secret_val = "vlt_p46_supersecret_token_99887766"
    secrets_provider.register_secret("p46_vault_demo", secret_val, allowed_services={"rag-gateway"})

    ev = audit_logger.log_event(
        event_type="p46_secret_redaction_test",
        summary=f"Downstream auth used {secret_val} against vault cluster",
    )
    assert secret_val not in ev["summary"], "Managed secret leaked into SHA-256 audit ledger summary!"
    assert "[REDACTED_SECRET]" in ev["summary"]
    assert audit_logger.verify_chain() is True


def test_p4_6_gradio_tab4_exposes_microservices_ops_controls():
    frontend_src = (REPO_ROOT / "frontend" / "app.py").read_text(encoding="utf-8")
    for token in (
        "/health/resilience",
        "/discovery/services",
        "/cqrs/read-model",
        "/ops/feature-flags",
        "/ops/chaos",
        "handle_update_feature_flag_ui",
        "handle_update_chaos_ui",
    ):
        assert token in frontend_src, f"frontend/app.py missing P4 Ops UI token: {token}"
