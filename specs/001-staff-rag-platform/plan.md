# Technical Architecture Plan & Module Blueprint

> **Spec-Kit Artifact:** `specs/001-staff-rag-platform/plan.md`
> **Companion Graphs:** Interactive AST knowledge graph in `graphify-out/graph.html` (`1,380` nodes, `2,554` edges, `103` communities).

---

## 1. End-to-End System Architecture

```mermaid
flowchart TB
    subgraph Clients["Clients & Peer Agents"]
        UI["Gradio 4-Tab Studio (:7860)"]
        MCP["MCP Clients (backend/mcp_server.py)"]
        A2A["Google A2A Peer Agents (/.well-known/agent.json, POST /a2a)"]
    end

    subgraph Gateway["FastAPI Microservices Gateway (:8000)"]
        MW["APIGatewayMiddleware\n(Rate Limit 429, X-Request-Id,\nX-Release-Track, Security Headers)"]
        ZT["ZeroTrustTokenAuthority\n(HS256 <=300s JWT + jti Replay Cache)"]
        DISC["ServiceRegistry\n(TTL Heartbeats & Stale Eviction)"]
        OPS["Progressive Delivery & Chaos\n(/ops/feature-flags, /ops/chaos)"]
    end

    subgraph Orchestration["Agentic & Multi-Agent Orchestration"]
        LG["Cyclic LangGraph StateGraph\n(Router -> Retrieve -> CRAG -> Web -> Synth -> Self-RAG)"]
        SUP["SupervisorOrchestrator\n(Parallel Specialist Subagents)"]
        STRUCT["Structured Taxonomy Engine\n(DoorDash Inverted RAG / Ramp JSON)"]
    end

    subgraph Retrieval["Tri-Modal Hybrid Retrieval & Caching"]
        CACHE["Semantic Cache + Labeled Sidecar\n+ Shared-Prefix KV Cache"]
        HYB["HybridRetrievalEngine\n(Pre-Filtered RBAC + Tenant Shards\n-> Dense + BM25 + Entity Graph\n-> RRF k=60 + Freshness/Version\n-> Cross-Encoder + MMR + Lost-in-Middle)"]
        SAGA["Ingestion & GDPR Sagas\n(Compensating Rollbacks + CQRS Read Model)"]
    end

    subgraph DataPlane["Owned Datastores & External Dependencies"]
        MIL["Milvus / Local Dense Vectors\n[Bulkhead: milvus_vector]"]
        SQL["SQLite OpsBot DB (Read-Only)\n[Bulkhead: sql_opsbot]"]
        OLL["Ollama Local LLM\n[Bulkhead: ollama_llm + Breaker]"]
        WEB["DuckDuckGo Search\n[Bulkhead: web_search + Breaker]"]
        RMQ["RabbitMQ Async Workers\n(with Thread Fallback)"]
    end

    subgraph Governance["Observability, Evaluation & Compliance"]
        AUD["SHA-256 Audit Ledger\n(PII + Secret Auto-Redaction)"]
        TR["Langfuse-Compatible TraceCollector\n(Hierarchical Span Waterfalls)"]
        PROM["Prometheus (:9091) -> Grafana (:3000, 8 Panels)"]
        EVAL["5-Mode IR Ablation & LLM-as-a-Judge"]
    end

    Clients --> MW --> ZT
    MW --> LG & SUP & STRUCT & SAGA & OPS & DISC
    LG & SUP & STRUCT --> CACHE --> HYB
    HYB --> MIL
    LG & SUP --> OLL & WEB & SQL
    SAGA --> HYB & AUD
    Gateway --> RMQ
    Gateway --> AUD & TR & PROM & EVAL
```

---

## 2. Cyclic LangGraph CRAG + Self-RAG Execution Flow

```mermaid
stateDiagram-v2
    [*] --> analyze_and_expand_query
    analyze_and_expand_query --> retrieve_and_rerank : Guardrails + Coreference + Adaptive Route
    retrieve_and_rerank --> grade_retrieval_crag : 8-Stage Funnel + Lost-in-the-Middle
    grade_retrieval_crag --> external_web_search : insufficient_web_fallback OR web enabled
    grade_retrieval_crag --> synthesize_cited_answer : sufficient (prob >= 0.40) & web disabled
    external_web_search --> synthesize_cited_answer : Bulkhead + Retry + Breaker + Chaos
    synthesize_cited_answer --> verify_faithfulness_self_rag : Prefix Cache + SQL OpsBot + Ollama/Extractive
    verify_faithfulness_self_rag --> analyze_and_expand_query : low faithfulness & retries < max_retries
    verify_faithfulness_self_rag --> [*] : grounded / revised / max_retries reached
```

---

## 3. Complete Repository Module Blueprint

### Backend Application Packages (`backend/app/`) — All 14 Packages

| Package | Modules | Responsibility |
| :--- | :--- | :--- |
| `backend/app/a2a/` | `server.py` | Google A2A Agent Card (`/.well-known/agent.json`), JSON-RPC 2.0 (`tasks/send`, `tasks/get`, `tasks/cancel`), SSE (`tasks/sendSubscribe`), and Anti-Confused-Deputy `PEER_AGENT_ROLE_CEILINGS`. |
| `backend/app/agent/` | `graph.py`, `nodes.py`, `state.py`, `sql_tool.py`, `structured.py`, `subagents.py` | Cyclic LangGraph CRAG/Self-RAG pipeline (wired to bulkheads, retries & chaos), Agoda read-only SQLite OpsBot, DoorDash/Ramp structured taxonomy classifier, and 4-specialist parallel `SupervisorOrchestrator`. |
| `backend/app/data/` | `saga.py`, `cqrs.py` | `SagaOrchestrator`, `execute_ingestion_saga`, `execute_gdpr_erasure_saga` with reverse compensating transactions, `CQRSReadProjection` O(1) read snapshots, and `DATABASE_PER_SERVICE_BOUNDARIES`. |
| `backend/app/evaluation/` | `benchmark.py`, `metrics.py`, `judge.py` | Golden QA 5-mode IR ablation suite (`HitRate@K`, `MRR@K`, `NDCG@K`, `ContextPrecision@K`) and `RAGJudgeEvaluator` (Faithfulness, Relevance, Dropbox filename/citation hallucination audit, 4-case human calibration). |
| `backend/app/gateway/` | `discovery.py`, `middleware.py` | Dynamic `ServiceRegistry` (TTL heartbeats & stale-instance eviction) and `APIGatewayMiddleware` (per-client sliding-window `429 Retry-After` rate limiting, `X-Request-Id`, `X-Release-Track`, security headers). |
| `backend/app/guardrails/` | `security.py`, `audit.py` | OWASP LLM01/LLM02 PII & prompt-injection sanitizer plus `TamperEvidentAuditLogger` (SHA-256 hash chain with automatic PII and managed-secret redaction). |
| `backend/app/ingestion/` | `chunking.py`, `pdf_parser.py`, `pipeline.py` | Structure-aware Parent-Child chunking (tables, fenced code, Python AST + polyglot symbols, Anthropic contextual prefix, optional Ollama contextualizer), NVIDIA layout/table PDF parser, SHA-256 deduplication, and CDC upsert/delete orchestration. |
| `backend/app/memory/` | `conversation.py` | Session-scoped multi-turn conversational history and deterministic pronoun coreference resolution (`it`, `that`, `they` -> prior subject). |
| `backend/app/observability/` | `metrics.py`, `tracing.py`, `feedback.py` | Rolling `p50`/`p95`/`p99` latency & Prometheus exposition (including feedback, prefix cache & bulkhead gauges), `CircuitBreaker`, Langfuse-compatible `TraceCollector` with hierarchical ASCII span waterfalls, and Zendesk-style `FeedbackStore`. |
| `backend/app/resilience/` | `patterns.py`, `progressive_delivery.py`, `chaos.py` | `retry_with_backoff` (exponential backoff + jitter), `BulkheadPool` / `BulkheadRegistry`, `evaluate_degradation_level` (`L0–L3`), `FeatureFlagManager` (deterministic Canary `%` & Blue-Green track), and gated `ChaosController`. |
| `backend/app/retrieval/` | `embeddings.py`, `entity_graph.py`, `hybrid.py`, `labeled_queries.py`, `postprocessor.py`, `prefix_cache.py`, `reranker.py`, `semantic_cache.py`, `semantic_ids.py` | Thread-safe `ModelRegistry`, 1-hop Entity Knowledge Graph, Tri-Modal `HybridRetrievalEngine` (Pre-Filtered RBAC, Airbnb tenant shards, Spotify 4-level Semantic IDs, Perplexity Late Chunking, RRF `k=60`, freshness/version boost), calibrated Cross-Encoder, MMR + Lost-in-the-Middle postprocessor, Semantic/Labeled/Shared-Prefix caches. |
| `backend/app/security/` | `zero_trust.py` | Pure-stdlib HS256 `ZeroTrustTokenAuthority` (`ttl_sec <= 300s`, `iss`/`aud`/`exp` validation, `jti` replay cache) and least-privilege `SecretsProvider` with per-service allow-lists and `redact_secrets()`. |
| `backend/app/worker/` | `consumer.py` | RabbitMQ async query (`rag_jobs`) and ingestion (`rag_ingestion_jobs`) workers with automatic local thread-pool fallback when RabbitMQ is offline. |
| `backend/app/` (root modules) | `config.py`, `models.py` | Environment-backed `RAGSettings` and all 21 typed Pydantic v2 request/response/trace data contracts. |

### Top-Level Platform Directories

| Path | Purpose |
| :--- | :--- |
| `backend/main.py` | FastAPI gateway exposing 46 HTTP/SSE/JSON-RPC endpoints and mounting `APIGatewayMiddleware`. |
| `backend/mcp_server.py` | RBAC-scoped MCP server exposing 5 read/write/compliance tools with zero tool-poisoning surface. |
| `backend/debug.py` | CLI diagnostic runner for local corpus seeding, retrieval inspection, and 5-mode ablation benchmarks. |
| `backend/tests/` | Automated TDD, Security/OWASP, IR Ablation, Subagents, Google A2A, Microservices, Release Polish & Spec-Kit integrity test suite (`89` tests). |
| `frontend/app.py` | 4-Tab Gradio Studio (Ingestion/PDF/CDC, SSE & Multi-Agent Query Studio, 5-Mode Eval & Judge, Compliance Audit & P4 Microservices Ops). |
| `deploy/mesh/` | Istio `STRICT` mTLS `PeerAuthentication` and `DestinationRule` retry/outlier-detection traffic manifests. |
| `deploy/k8s/` | Immutable Kubernetes `Deployment` (`readOnlyRootFilesystem`, `runAsNonRoot`, digest-pinned), `Service`, `HPA`, `PDB`, and zero-trust `NetworkPolicy`. |
| `deploy/terraform/` | Immutable Terraform IaC skeleton managing cluster namespace, release-track ConfigMap, and digest pinning. |
| `observability/` | Prometheus scrape config (`prometheus.yml`) and pre-provisioned Grafana datasource & 8-panel dashboard (`rag_observability.json`). |
| `scripts/docker_stack_smoke_test.py` | Full-stack live integration smoke test verifying all 7 Docker Compose service endpoints end-to-end. |
| `.github/workflows/edd.yml` | GitHub Actions Eval-Driven Development gate running syntax checks, full pytest suite, and 5-mode IR regression assertions. |
