"""TDD tests for P4.2: Dynamic Service Discovery, API Gateway Rate Limiting/Headers, and Istio Service Mesh mTLS."""

from pathlib import Path
import time
from fastapi.testclient import TestClient

from main import app
from app.gateway.discovery import ServiceRegistry
from app.gateway.middleware import GatewayRateLimiter


client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dynamic_service_discovery_registers_resolves_and_evicts_stale_instances():
    """ServiceRegistry tracks dynamic instances, round-robins healthy targets, and evicts expired heartbeats."""
    reg = ServiceRegistry(seed_defaults=False)
    reg.register("reranker-svc", "inst-1", "10.0.1.11", 8081, ttl_sec=0.05)
    reg.register("reranker-svc", "inst-2", "10.0.1.12", 8081, ttl_sec=5.0)

    first = reg.resolve("reranker-svc")
    second = reg.resolve("reranker-svc")
    assert {first["instance_id"], second["instance_id"]} == {"inst-1", "inst-2"}

    time.sleep(0.08)
    evicted = reg.prune_expired()
    assert evicted == 1
    remaining = reg.list_services()["services"]["reranker-svc"]
    assert len(remaining) == 1
    assert remaining[0]["instance_id"] == "inst-2"

    # Live HTTP discovery registration & listing
    r_reg = client.post(
        "/discovery/register",
        json={"service_name": "embedding-worker", "instance_id": "emb-01", "host": "10.0.2.5", "port": 8090, "ttl_sec": 30},
    )
    assert r_reg.status_code == 200
    r_list = client.get("/discovery/services")
    assert r_list.status_code == 200
    assert "embedding-worker" in r_list.json()["services"]


def test_api_gateway_middleware_headers_and_token_bucket_rate_limiting():
    """API Gateway injects X-Request-Id & security headers and enforces 429 rate limits on bursts."""
    resp = client.get("/health", headers={"X-Request-Id": "req-gw-12345"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-Id") == "req-gw-12345"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Gateway-Service") == "staff-rag-gateway"

    limiter = GatewayRateLimiter(max_requests_per_window=2, window_sec=60.0)
    assert limiter.allow("tenant_burst")[0] is True
    assert limiter.allow("tenant_burst")[0] is True
    allowed, retry_after = limiter.allow("tenant_burst")
    assert allowed is False
    assert retry_after >= 1


def test_istio_service_mesh_mtls_and_traffic_manifests_present():
    """Verifies deploy/mesh/istio-mtls-and-traffic.yaml configures STRICT mTLS, outlier detection, and retries."""
    mesh_file = REPO_ROOT / "deploy" / "mesh" / "istio-mtls-and-traffic.yaml"
    assert mesh_file.exists()
    text = mesh_file.read_text(encoding="utf-8")
    assert "kind: PeerAuthentication" in text
    assert "mode: STRICT" in text
    assert "kind: DestinationRule" in text
    assert "ISTIO_MUTUAL" in text
    assert "outlierDetection:" in text
    assert "kind: VirtualService" in text
