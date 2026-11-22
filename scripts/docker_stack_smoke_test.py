"""End-to-End Live Docker Compose Stack Integration & Smoke Test.

Verifies the full running microservices stack:
- FastAPI Gateway (:8000): /health, /health/resilience, /ingest-sample, /query-stream, /metrics/prometheus
- Gradio 4-Tab Studio (:7860)
- Milvus Standalone (:19530 TCP)
- RabbitMQ Management (:15672)
- Ollama LLM Server (:11434)
- Prometheus (:9091) & Grafana (:3000)
"""

from __future__ import annotations

import os
import socket
import sys
import time
from typing import Dict
import requests


def wait_http(url: str, timeout_sec: float = 90.0) -> requests.Response:
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code < 500:
                return resp
        except Exception as exc:
            last_err = exc
        time.sleep(2.0)
    raise RuntimeError(f"Service at {url} did not become ready within {timeout_sec}s: {last_err}")


def check_tcp(host: str, port: int, timeout_sec: float = 5.0) -> bool:
    with socket.create_connection((host, port), timeout=timeout_sec):
        return True


def run_smoke_test() -> Dict[str, object]:
    api = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    ui = os.getenv("FRONTEND_URL", "http://127.0.0.1:7860").rstrip("/")
    prom = os.getenv("PROMETHEUS_URL", "http://127.0.0.1:9091").rstrip("/")
    graf = os.getenv("GRAFANA_URL", "http://127.0.0.1:3000").rstrip("/")
    rmq = os.getenv("RABBITMQ_MGMT_URL", "http://127.0.0.1:15672").rstrip("/")
    ollama = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
    milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

    # 1. Core API & Resilience Health
    r_health = wait_http(f"{api}/health/resilience", timeout_sec=120.0)
    assert r_health.status_code == 200, r_health.text
    res_data = r_health.json()

    # 2. Seed sample corpus & execute live SSE RAG query
    r_seed = requests.post(f"{api}/ingest-sample", timeout=45)
    assert r_seed.status_code == 200, r_seed.text

    r_sse = requests.post(
        f"{api}/query-stream",
        json={
            "text": "What triggers ERR_LEDGER_9042?",
            "retrieval_mode": "hybrid_graph_rerank",
            "top_k": 3,
            "tenant_id": "default",
            "user_roles": ["public", "engineering"],
            "enable_web_search": False,
            "enable_semantic_cache": False,
        },
        timeout=60,
    )
    assert r_sse.status_code == 200 and '"event": "final"' in r_sse.text

    # 3. Verify Prometheus exposition & scraper + Grafana + RabbitMQ + Ollama + Milvus + Gradio UI
    r_prom_exp = requests.get(f"{api}/metrics/prometheus", timeout=10)
    assert r_prom_exp.status_code == 200 and "rag_bulkhead_saturation_ratio" in r_prom_exp.text

    assert wait_http(ui, timeout_sec=60.0).status_code == 200
    assert wait_http(f"{prom}/-/ready", timeout_sec=60.0).status_code == 200
    assert wait_http(f"{graf}/api/health", timeout_sec=60.0).status_code == 200
    assert wait_http(rmq, timeout_sec=60.0).status_code == 200
    assert wait_http(f"{ollama}/api/tags", timeout_sec=60.0).status_code == 200
    assert check_tcp(milvus_host, milvus_port, timeout_sec=10.0) is True

    return {
        "status": "PASSED",
        "degradation_level": res_data.get("degradation", {}).get("level"),
        "milvus_connected": res_data.get("degradation", {}).get("milvus_connected"),
        "services_verified": [
            "fastapi-backend:8000",
            "frontend:7860",
            "milvus:19530",
            "rabbitmq:15672",
            "ollama:11434",
            "prometheus:9091",
            "grafana:3000",
        ],
    }


if __name__ == "__main__":
    try:
        summary = run_smoke_test()
        print("DOCKER STACK LIVE SMOKE TEST PASSED:", summary)
    except Exception as exc:
        print(f"DOCKER STACK LIVE SMOKE TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
