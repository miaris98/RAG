# API, MCP, Google A2A & Telemetry Contract Reference

> **Spec-Kit Artifact:** `specs/001-staff-rag-platform/contracts/api-and-protocols.md`
> **Auto-Enforced Zero Drift:** Verified against live `main.app.routes`, `MCP_TOOL_MANIFEST`, and `build_agent_card()` by `backend/tests/test_docs_spec_kit_completeness.py`.

---

## 1. Cross-Cutting API Gateway Headers (`APIGatewayMiddleware`)

Every HTTP response from `backend/main.py` includes:
- `X-Request-Id`: Echoed from caller or generated (`req_<12hex>`).
- `X-Gateway-Service`: `staff-rag-gateway`.
- `X-Release-Track`: Active Blue-Green / Canary release track (`blue` | `green` | `canary`).
- `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY`.
- Rate limiting per `X-Tenant-Id` / `X-Peer-Agent-Id` / client IP returns HTTP `429` with `Retry-After`.

---

## 2. Complete FastAPI HTTP Endpoint Catalog (46 Endpoints)

### Ingestion, Corpus & CDC
| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/ingest` | Chunks, sanitizes (PII/injection), deduplicates (SHA-256), and indexes a document across Dense, BM25, and Entity Graph. |
| `POST` | `/ingest-async` | Enqueues an asynchronous document ingestion job into RabbitMQ (with thread fallback) and returns `job_id`. |
| `POST` | `/ingest/pdf` | NVIDIA layout- & table-aware PDF parser + hybrid ingestion (`multipart/form-data`). |
| `POST` | `/ingest/cdc` | Shopify/Notion-style CDC incremental `upsert` / `delete` keyed by `external_id` with version precedence. |
| `POST` | `/ingest-sample` | Seeds the 3 built-in Enterprise Architecture & Governance sample documents. |
| `GET` | `/corpus` | Returns active document, parent-window, and child-chunk statistics. |
| `DELETE` | `/corpus` | Clears in-memory hybrid indices and semantic cache for clean benchmark resets. |
| `GET` | `/corpus/shards-and-clusters` | Inspects Airbnb tenant partition shard topology and Spotify `sem.*` Semantic ID clusters. |
| `GET` | `/documents` | Lists per-document inventory, tenant/RBAC role lineage, and chunk counts. |
| `DELETE` | `/documents/{doc_id}` | GDPR Article 17 surgical erasure across all indices, caches, and CQRS projections + immutable audit event. |

### Query, Retrieval, Caching & Structured Output
| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/retrieve` | Executes Pre-Filtered RBAC + Tri-Modal Hybrid Retrieval (with optional `semantic_id_prefix`) without LLM generation. |
| `POST` | `/query-stream` | Streams live LangGraph node progression and final cited response via Server-Sent Events (`text/event-stream`). |
| `POST` | `/submit-job` | Enqueues an asynchronous RAG query job into RabbitMQ (`rag_jobs`) with local thread fallback. |
| `GET` | `/get-job/{job_id}` | Polls status, grounded answer, tools used, and `PipelineTrace` for a submitted job. |
| `POST` | `/query-structured` | DoorDash Inverted RAG constrained taxonomy enum classification + Ramp rule-grounded JSON with citations. |
| `POST` | `/cache/prewarm` | Instacart-style offline head-query pre-warming into role-scoped Semantic & Prefix caches. |
| `GET` | `/cache/stats` | Returns Semantic Query Cache and LinkedIn Shared-Prefix KV Prompt Cache statistics. |

### Live SQL OpsBot, Multi-Agent Subagents & Google A2A
| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/inventory/sql-status` | Agoda OpsBot live operational SQLite service inventory & active incidents snapshot. |
| `POST` | `/inventory/sql-query` | Strictly read-only `SELECT` query execution against operational SQLite (mutations return `400`). |
| `GET` | `/agent/subagents` | Lists the 4 registered least-privilege Specialist Subagents and their allowed tool scopes. |
| `POST` | `/agent/orchestrate` | Dispatches parallel Specialist Subagents under `SupervisorOrchestrator` with RBAC isolation & span waterfall. |
| `GET` | `/.well-known/agent.json` | Google A2A Agent Card discovery manifest advertising skills, streaming support, and DelegatedRBAC auth. |
| `POST` | `/a2a` | Google A2A JSON-RPC 2.0 (`tasks/send`, `tasks/get`, `tasks/cancel`) & SSE (`tasks/sendSubscribe`); verifies `X-Service-Token` JWT (`401` on invalid/replay) when supplied and enforces `X-Peer-Agent-Id` anti-confused-deputy ceilings. |

### Observability, Evaluation, Feedback & Compliance
| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health` | System readiness, loaded singleton model status, and hybrid index statistics. |
| `GET` | `/metrics` | Rolling `p50`/`p95`/`p99` latencies, cache hit rate, guardrail triggers, and circuit breaker states. |
| `GET` | `/metrics/prometheus` | Raw Prometheus text exposition (`text/plain; version=0.0.4`) scraped every 5s. |
| `GET` | `/traces` | Lists recent Langfuse-compatible distributed RAG traces and span timings. |
| `GET` | `/traces/{trace_id}` | Retrieves a single trace and hierarchical span waterfall by `trace_id`. |
| `GET` | `/audit-logs` | Returns tamper-evident SHA-256 hash-chained audit events and `chain_valid` boolean. |
| `GET` | `/compliance/audit-report` | Generates full Executive Compliance & Observability Dossier (JSON + Markdown). |
| `POST` | `/feedback` | Records Zendesk-style thumbs-up/down feedback linked to `trace_id` and `chunk_ids`. |
| `GET` | `/feedback` | Returns rolling satisfaction rate and recent feedback entries. |
| `POST` | `/evaluate` | Runs 5-mode Golden QA IR Ablation Benchmark + `RAGJudgeEvaluator` & human calibration. |

### P4 Resilient Microservices Control Plane
| Method | Path | Purpose |
| :--- | :--- | :--- |
| `GET` | `/health/resilience` | Graceful Degradation Matrix (`L0–L3`), per-pool Bulkhead saturation, Circuit Breakers, and retry telemetry. |
| `POST` | `/discovery/register` | Dynamic Service Discovery instance registration with TTL. |
| `POST` | `/discovery/heartbeat/{service_name}/{instance_id}` | Refreshes TTL heartbeat for a registered microservice instance. |
| `GET` | `/discovery/services` | Lists live healthy instances (after stale TTL eviction) and API Gateway rate-limiter stats. |
| `POST` | `/saga/ingest` | Executes multi-step Ingestion Saga with automatic reverse compensating rollbacks on failure. |
| `GET` | `/cqrs/read-model` | Returns lock-free O(1) materialized CQRS document/tenant/role read projection. |
| `GET` | `/data/boundaries` | Exposes Database-per-Service ownership contracts (`shared_db_forbidden: True`). |
| `POST` | `/security/token` | Issues short-lived (`<= 300s`) Zero Trust HS256 service JWT with unique `jti`. |
| `POST` | `/security/verify` | Validates `X-Service-Token` JWT signature, `iss`/`aud`/`exp`, and one-time `jti` replay cache (`401` on failure). |
| `GET` | `/ops/feature-flags` | Returns Canary percentage feature flags, deterministic evaluation for `subject_key`, and active Blue-Green track. |
| `POST` | `/ops/feature-flags` | Updates a Canary rollout percentage (`0–100%`) and/or switches `active_release_track` (`blue`/`green`/`canary`). |
| `GET` | `/ops/chaos` | Returns gated Chaos Engineering controller status and injected fault/latency counters. |
| `POST` | `/ops/chaos` | Configures controlled synthetic failure or latency injection for `ollama`, `duckduckgo`, `milvus`, `rabbitmq`, or `sql_opsbot`. |

---

## 3. MCP Server Contract (`backend/mcp_server.py` — 5 Tools)

All MCP tool descriptions are red-team scanned by `test_security_owasp_and_mcp.py`:
1. `retrieve_hybrid` — Read-only pre-filtered RBAC Tri-Modal Hybrid Retrieval + Cross-Encoder reranking.
2. `ingest_document` — PII/injection-sanitized, SHA-256 deduplicated ingestion into vector/lexical/graph indices.
3. `gdpr_erase_document` — GDPR Article 17 surgical deletion across stores + immutable audit log.
4. `query_live_sql_inventory` — Validated read-only `SELECT` against operational SQLite (`service_inventory`, `active_incidents`).
5. `classify_structured_taxonomy` — Read-only constrained taxonomy enum classification grounded in RBAC-authorized RAG citations.

---

## 4. Google A2A Protocol Contract (`backend/app/a2a/server.py`)

- **Discovery Card:** `GET /.well-known/agent.json` (`capabilities: {streaming: true, pushNotifications: false, stateTransitionHistory: true}`).
- **Advertised Skills (4):**
  - `enterprise_rag_qa` — Enterprise Grounded RAG Q&A
  - `live_sql_ops_audit` — Live SQL OpsBot Inventory & Incidents
  - `taxonomy_rule_classifier` — Constrained Taxonomy & Rule-Grounded JSON Classifier
  - `multi_agent_research` — Parallel Specialist Subagent Supervisor
- **JSON-RPC 2.0 Methods:** `tasks/send`, `tasks/get`, `tasks/cancel`, `tasks/sendSubscribe` (SSE).
- **Security Controls:** Optional/strict `X-Service-Token` HS256 JWT + `jti` replay rejection (`401`) plus `X-Peer-Agent-Id` anti-confused-deputy role ceilings (`PEER_AGENT_ROLE_CEILINGS`).

---

## 5. Prometheus Exposition Metrics (`GET /metrics/prometheus`)

`rag_queries_total`, `rag_cache_hits_total`, `rag_crag_fallbacks_total`, `rag_self_rag_retries_total`, `rag_guardrail_triggers_total`, `rag_gdpr_erasures_total`, `rag_latency_ms_p50`, `rag_latency_ms_p95`, `rag_latency_ms_p99`, `rag_corpus_documents_total`, `rag_corpus_chunks_total`, `rag_traces_total`, `rag_audit_chain_valid`, `rag_circuit_breaker_open{service}`, `rag_user_feedback_total{rating}`, `rag_user_satisfaction_rate`, `rag_prefix_cache_hits_total`, `rag_prefix_cache_amortized_tokens_total`, `rag_bulkhead_saturation_ratio{pool}`, `rag_bulkhead_rejected_total{pool}`.
