"""TDD tests for Release Polish: Grafana Panel 8 (P4 Bulkheads), Docker Compose Hardening & CI Frontend Smoke."""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_grafana_dashboard_includes_p4_bulkhead_saturation_and_rejection_panel():
    dash_path = REPO_ROOT / "observability" / "grafana" / "dashboards" / "rag_observability.json"
    data = json.loads(dash_path.read_text(encoding="utf-8"))
    panels = data.get("panels") or []
    assert len(panels) >= 8, f"Expected >= 8 Grafana panels (including P4 bulkheads), found {len(panels)}"

    exprs = [
        t.get("expr", "")
        for p in panels
        for t in (p.get("targets") or [])
    ]
    assert any("rag_bulkhead_saturation_ratio" in e for e in exprs), "Missing rag_bulkhead_saturation_ratio panel"
    assert any("rag_bulkhead_rejected_total" in e for e in exprs), "Missing rag_bulkhead_rejected_total panel"


def test_docker_compose_hardened_with_healthcheck_and_zero_trust_secret():
    compose_text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for required in (
        "ZERO_TRUST_JWT_SECRET",
        "/health/resilience",
        "milvus-standalone",
        "rabbitmq",
        "ollama",
        "prometheus",
        "grafana",
    ):
        assert required in compose_text, f"docker-compose.yml missing '{required}'"


def test_ci_workflow_validates_frontend_app_compilation():
    ci_text = (REPO_ROOT / ".github" / "workflows" / "edd.yml").read_text(encoding="utf-8")
    assert "frontend/app.py" in ci_text, ".github/workflows/edd.yml missing frontend/app.py smoke check"


def test_docker_stack_live_smoke_script_present_and_compiles():
    smoke_script = REPO_ROOT / "scripts" / "docker_stack_smoke_test.py"
    assert smoke_script.exists(), "Missing scripts/docker_stack_smoke_test.py"
    src = smoke_script.read_text(encoding="utf-8")
    compile(src, str(smoke_script), "exec")
    for required in ("/health/resilience", "/ingest-sample", "/metrics/prometheus", "15672", "9091", "3000"):
        assert required in src, f"Smoke script missing check for '{required}'"
