# Deployment, Observability, Chaos & Security Operations Runbook

> **Spec-Kit Artifact:** `specs/001-staff-rag-platform/operations-runbook.md`

---

## 1. Runtime Environments & Deployment Topologies

### A. Fast Local Mode (`.venv` — Recommended for Development & Live Demos)
```powershell
# Terminal 1 — FastAPI Gateway (:8000)
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Gradio 4-Tab Studio (:7860)
cd frontend
$env:BACKEND_URL = "http://127.0.0.1:8000"
..\.venv\Scripts\python.exe app.py
```

### B. Full Docker Compose Stack (9 Microservices)
Services: `fastapi-backend` (`:8000`, healthchecked at `/health/resilience`), `frontend` (`:7860`), `milvus` (`:19530` + `etcd` + `minio`), `attu` (`:9090`), `rabbitmq` (`:5672`, `:15672`), `ollama` (`:11434`), `prometheus` (`:9091`), `grafana` (`:3000`).
```powershell
docker-compose config --quiet
docker-compose up -d --build
docker exec -it ollama ollama pull llama3.1
.\.venv\Scripts\python.exe scripts\docker_stack_smoke_test.py
```

### C. Immutable Kubernetes, Service Mesh & Terraform IaC (`deploy/`)
- **Kubernetes (`deploy/k8s/rag-deployment-hpa-netpol.yaml`):** Digest-pinned immutable `Deployment` (`readOnlyRootFilesystem: true`, `runAsNonRoot: true`, dropped Linux capabilities, readiness probe on `/health/resilience`, liveness probe on `/health`), `Service`, `HorizontalPodAutoscaler` (`3..12` replicas @ 65% CPU), `PodDisruptionBudget` (`minAvailable: 2`), and zero-trust ingress/egress `NetworkPolicy`.
- **Istio Service Mesh (`deploy/mesh/istio-mtls-and-traffic.yaml`):** `PeerAuthentication` (`STRICT` mTLS across namespace) + `DestinationRule` connection pools & outlier eviction.
- **Terraform (`deploy/terraform/main.tf`):** Declarative namespace (`istio-injection=enabled`), digest-pinned release config, and Blue-Green track variable.

---

## 2. Observability & Alerting Runbook

- **Grafana (`:3000`):** Pre-provisioned `Staff+ RAG Observability & Audit Dashboard` (`observability/grafana/dashboards/rag_observability.json`) with **8 panels**:
  1. Latency Percentiles (`p50` / `p95` / `p99`)
  2. Queries, Cache Hits, Guardrails & GDPR Erasures
  3. Active Corpus Documents & Chunks
  4. SHA-256 Audit Chain Integrity (`1 = VALID`, `0 = TAMPERED`)
  5. Circuit Breakers Open (`ollama_llm`, `duckduckgo_web`)
  6. User Feedback Ratings & Satisfaction Rate
  7. Shared-Prefix KV Cache Hits & Amortized Tokens
  8. P4 Bulkhead Compartment Saturation & Rejections (`milvus_vector`, `ollama_llm`, `sql_opsbot`, `web_search`)
- **Gradio Tab 4 (`Compliance Audit & Observability Center`):** One-click live view of the Compliance Dossier, Feedback KPIs, Airbnb Shards & Spotify Clusters, Prefix/Semantic Cache stats, P4 Resilience/Discovery/CQRS/Canary/Chaos status, Agoda SQL Inspector, Distributed Trace Span Waterfalls, SHA-256 Audit Ledger, and raw Prometheus exposition.

---

## 3. Graceful Degradation Matrix (`GET /health/resilience`)

| Tier | Trigger Condition | System Behavior | Operator Action |
| :--- | :--- | :--- | :--- |
| `L0_FULL_OPERATIONAL` | All breakers `CLOSED`, Milvus connected, bulkheads healthy | Full Tri-Modal GraphRAG + Ollama LLM + Web Search | Normal operation |
| `L1_EXTRACTIVE_DEGRADED` | `ollama_llm` or `duckduckgo_web` Circuit Breaker `OPEN` | Deterministic citation-grounded extractive synthesis (`core_rag_available: true`) | Check Ollama container / GPU memory or disable active Chaos fault |
| `L2_LOCAL_HYBRID_FALLBACK` | External Milvus disconnected (`milvus_connected: false`) | Full Dense + BM25 + Entity Graph served from local hybrid tier | Inspect `milvus-standalone`, `etcd`, and `minio` containers |
| `L3_CACHE_ONLY_SURVIVAL` | `>= 2` Bulkhead pools at `100%` saturation | Shed excess concurrency (`BulkheadRejectedError`), serve Semantic & Labeled caches | Scale replicas via HPA (`maxReplicas: 12`) or tighten Gateway rate limits |

---

## 4. Progressive Delivery & Gated Chaos SOP

### Canary Rollouts & Blue-Green Track Switching
1. Inspect active flags: `GET /ops/feature-flags?subject_key=<tenant_or_user>` (check `X-Release-Track` response header).
2. Ramp Canary deterministically (`5% -> 25% -> 100%`) or cut over Blue-Green track (`blue` -> `green`) via `POST /ops/feature-flags` or Tab 4's **P4 Microservices Ops** accordion.
3. Verify error rate, latency `p95`, and `rag_user_satisfaction_rate` before promoting to `100%`; instant rollback sets `active_release_track: "blue"` or `rollout_percentage: 0`.

### Gated Chaos Engineering Game-Day Procedure
1. Chaos is **OFF by default**. Enable strictly during game-day or automated tests via `POST /ops/chaos` (e.g., `target="ollama", mode="failure", failure_rate=1.0`).
2. Send a live query via `POST /query-stream` and confirm:
   - `milvus_vector` & `ollama_llm` bulkheads increment `total_acquired`.
   - `llm_circuit_breaker` trips `OPEN` and `GET /health/resilience` transitions to `L1_EXTRACTIVE_DEGRADED`.
   - End users still receive a grounded `[Doc i]` cited response.
3. **Mandatory Cleanup:** Reset chaos (`enabled: false, failure_rate: 0.0, latency_ms: 0`) and confirm `GET /ops/chaos` reports `enabled: false`.

---

## 5. Security, Zero Trust & GDPR Article 17 Procedures

- **Zero Trust Service Token Issuance & Verification:** Services request short-lived HS256 JWTs (`ttl_sec <= 300`) via `POST /security/token` and pass `X-Service-Token: Bearer <jwt>` on `POST /security/verify` or `POST /a2a`. Replayed `jti` values are immediately rejected (`401`).
- **GDPR Article 17 Erasure:** Call `DELETE /documents/{doc_id}` (or use Tab 4's **Erase Document** control). Confirm `chunks_deleted > 0`, `cache_entries_evicted >= 0`, `cqrs_read_model` omits `doc_id`, and `GET /audit-logs` shows `chain_valid: true` with a new `gdpr_article17_erasure` hash-chained record.
- **Audit Tamper Investigation:** If `rag_audit_chain_valid == 0` or `GET /audit-logs` returns `chain_valid: false`, treat as a P0 integrity incident, snapshot `/audit-logs`, and freeze write paths immediately.
