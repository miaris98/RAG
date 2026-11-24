# Staff+ RAG & Resilient Microservices Interview Guide

This guide is designed for **Senior / Staff / Principal Applied AI & Systems Design interviews**. Rather than focusing on mathematical formulas, it explains **why** the platform is designed the way it is, **how** the pieces fit together, the **engineering principles** enforced, the **tradeoffs and assumptions** made, and the **known gaps** to address for multi-replica cloud scale.

---

## 1. 90-Second Architecture Pitch

> *"I built a modular, 100% local-capable **Production RAG & Resilient Microservices Platform** using FastAPI, Milvus, RabbitMQ, LangGraph, Ollama, Gradio, Prometheus, and Grafana. Instead of a naive dense-only prototype, it uses **Pre-Filtered Multi-Tenant RBAC**, **Structure-Aware Parent-Child Chunking** with SHA-256 deduplication, and **Tri-Modal Hybrid Retrieval** (Dense + BM25 + 1-hop Entity Knowledge Graph) followed by calibrated Cross-Encoder reranking, MMR diversity, and Lost-in-the-Middle prompt ordering. Orchestration runs on an explicit **cyclic LangGraph StateGraph** (CRAG + Self-RAG) alongside **4 least-privilege Specialist Subagents**, **MCP**, and **Google A2A**. Finally, it is hardened with **P4 Resilient Microservices** patterns—bulkheads, bounded retries with jitter, `L0–L3` graceful degradation, dynamic service discovery, Saga compensating rollbacks, CQRS read projections, Zero Trust short-lived JWTs, Canary/Blue-Green feature flags, and gated Chaos Engineering."*

---

## 2. Solution Design: Why & How

### A. Ingestion & Knowledge Modeling (`backend/app/ingestion/`)
- **Why:** Naive fixed-size character splitting breaks Markdown tables, splits code blocks mid-function, and loses document context. Re-ingesting the same document creates duplicate vectors that crowd out top-$K$ results.
- **How:**
  - **Parent-Child Chunking:** Small child chunks are indexed for high-precision retrieval, while larger parent windows are promoted for LLM synthesis so the model has full surrounding context.
  - **Structure & Code Awareness:** Preserves Markdown tables and fenced code blocks, extracts Python AST and polyglot symbols, and prepends Anthropic-style contextual headers (with an optional Ollama LLM contextualizer).
  - **Idempotent SHA-256 Deduplication & CDC:** Deterministic content-addressable chunk IDs (`chk_<hash>`) skip duplicates automatically; `POST /ingest/cdc` handles incremental upserts/deletes and invalidates stale cache entries.
  - **PDF & Late Chunking:** NVIDIA-style layout/table PDF parsing (`POST /ingest/pdf`) and Perplexity-style late chunking boundary blending.

### B. Pre-Filtered RBAC & Tri-Modal Hybrid Retrieval (`backend/app/retrieval/`)
- **Why:** Post-retrieval RBAC filtering leaks existence/timing or causes **recall starvation** (returning 0 results when top-$K$ matches belong to another tenant). Dense-only embeddings miss exact identifiers (`ERR_LEDGER_9042`, `PD-EMEA-8821`), and uncalibrated top-$K$ similarity forces hallucinations when no relevant doc exists.
- **How:**
  - **Pre-Filtering First:** `tenant_id`, `access_roles`, Airbnb tenant shard routing, and Spotify 4-level Semantic IDs filter candidates **before** any vector, BM25, or graph scoring.
  - **Tri-Modal Fusion:** Dense (Milvus/local) + Sparse (`BM25Okapi`) + 1-hop Entity Knowledge Graph are fused via rank-based RRF, boosted by freshness time-decay and version precedence, then calibrated by a Cross-Encoder and pruned below `RELEVANCE_THRESHOLD=0.25`.
  - **Post-Processing:** MMR removes near-duplicate chunks, and Lost-in-the-Middle reordering places highest-confidence evidence at the start and end of the prompt.

### C. Explicit Cyclic Orchestration, Subagents & Protocols (`backend/app/agent/`, `backend/app/a2a/`, `backend/mcp_server.py`)
- **Why:** Open-ended ReAct loops are unpredictable with smaller local models—skipping tools, looping, or hallucinating arguments. Meanwhile, enterprise agents need least-privilege isolation and standard interoperability.
- **How:**
  - **Cyclic LangGraph `StateGraph`:** Guarantees bounded, instrumented execution (`Router -> Retrieve -> CRAG Gate -> [Web Search] -> Synthesize -> Self-RAG Faithfulness Check`), looping back only when faithfulness is low and retries remain.
  - **4 Least-Privilege Specialist Subagents:** `retrieval_specialist`, `sql_ops_specialist` (strictly read-only `SELECT` against operational SQLite), `compliance_auditor`, and `web_researcher` run in parallel under `SupervisorOrchestrator` with a unified span waterfall.
  - **MCP & Google A2A:** Exposes 5 RBAC-scoped MCP tools and a Google A2A server (`/.well-known/agent.json`, `POST /a2a`) with **Anti-Confused-Deputy** role ceilings (`PEER_AGENT_ROLE_CEILINGS`) so peer agents cannot escalate privileges.

### D. Resilient Microservices, Zero Trust & Compliance (`backend/app/resilience/`, `gateway/`, `data/`, `security/`, `guardrails/`)
- **Why:** In distributed AI systems, a slow LLM or search API can exhaust all worker threads and take down the entire gateway (cascading failure). Compliance requires provable auditability and surgical data erasure.
- **How:**
  - **Fault Isolation:** Per-dependency `BulkheadPool` compartments (`milvus_vector`, `ollama_llm`, `sql_opsbot`, `web_search`), bounded `retry_with_backoff` with jitter, and `CircuitBreaker` state machines drive an `L0–L3` **Graceful Degradation Matrix** (`GET /health/resilience`). Even if Ollama is down or under Chaos injection, core RAG degrades cleanly to citation-grounded extractive synthesis.
  - **Data & Traffic:** `APIGatewayMiddleware` enforces sliding-window rate limiting (`429 Retry-After`), request correlation (`X-Request-Id`), and Blue-Green/Canary release headers (`X-Release-Track`). `SagaOrchestrator` provides reverse compensating rollbacks on ingestion/GDPR failure, and `CQRSReadProjection` decouples `O(1)` reads from write stores.
  - **Zero Trust & GDPR:** Short-lived HS256 service JWTs (`ttl <= 300s`, `jti` replay cache) protect `/security/verify` and `/a2a`. `TamperEvidentAuditLogger` scrubs PII and managed secrets automatically and hash-chains every event with SHA-256; `DELETE /documents/{doc_id}` erases data across all stores and logs an immutable GDPR Article 17 record.

---

## 3. Core Engineering Principles Followed

Codified in [constitution.md](file:///c:/Users/miari/Desktop/RAG/.specify/memory/constitution.md) and enforced by automated tests:

1. **Pre-Filtered Security:** Authorization always precedes scoring; caches are strictly tenant- and role-scoped.
2. **Groundedness Over Fluency:** Every internal claim requires `[Doc i]` citations and evidence quotes; low-confidence retrieval triggers abstention or web fallback rather than hallucination.
3. **Idempotency Everywhere:** Content-addressable SHA-256 chunk IDs and CDC versioning make ingestion safe to retry.
4. **Least Privilege & Zero Trust:** Subagents, SQL OpsBot, secrets, and A2A peers operate under minimal scoped permissions and short-lived replay-protected tokens.
5. **Failure Isolation & Graceful Degradation:** Slow or failing external dependencies never crash core citation-grounded RAG.
6. **Database-per-Service & Compensating Sagas:** Services own their datastores behind APIs; multi-store writes roll back cleanly on failure.
7. **Eval-Driven & Test-Driven Development (EDD + TDD):** Every feature is backed by Red-Green-Verify tests (`89` tests) and a CI 5-mode IR regression gate.

---

## 4. Key Tradeoffs & Assumptions

| Decision | Alternative Considered | Why We Chose This | Key Assumption / Cost |
| :--- | :--- | :--- | :--- |
| **Explicit LangGraph `StateGraph`** | Open-ended `create_react_agent` loop | Deterministic DAG, bounded latency, per-node observability, reliable with local 8B models | Less ad-hoc flexibility for novel multi-step tool chains outside defined edges |
| **Pre-Filtered RBAC + Local Hybrid Fallback** | Post-retrieval ACL filtering | Eliminates data leakage and recall starvation; works even when Milvus is offline | Assumes corpus metadata (`tenant_id`, `access_roles`) is accurately tagged at ingestion |
| **In-Memory Singletons (Audit, CQRS, Discovery, Flags, Cache, `jti`)** | External Redis + PostgreSQL from day 1 | Zero external infra required for local `.venv` demos, fast CI (`<2 min`), deterministic tests | **Single-process assumption:** state is not shared across multiple horizontal replicas |
| **Pure-Stdlib HS256 JWT + Local `SecretsProvider`** | External Vault + OIDC IdP | Self-contained Zero Trust verification (`jti` replay, `<=300s` TTL) without external auth servers | Assumes symmetric secret rotation is managed via environment/Vault in production |
| **Rank-Based RRF + Cross-Encoder Calibration** | Linear score blending (`alpha * dense + beta * sparse`) | Scale-invariant across bounded cosine and unbounded BM25 scores | Adds ~15–30ms Cross-Encoder latency on top-$M$ candidates |
| **Deterministic Extractive Fallback on LLM Failure** | Returning HTTP 500 when Ollama is down | Preserves `core_rag_available: true` (`L1_EXTRACTIVE_DEGRADED`) under outages or Chaos | Fallback answers are extractive/structured rather than abstractive prose |

---

## 5. Honest Gaps & Production Roadmap (Beyond Local Showcase)

If asked *"What would you change before deploying this to multi-region production?"*, highlight these concrete gaps:

1. **Externalize In-Memory State for Multi-Replica HPA:**
   - Currently, the SHA-256 audit ledger, CQRS read model, Service Discovery registry, Feature Flags, Semantic Cache, and JWT `jti` replay cache live in process memory. Scaling `HorizontalPodAutoscaler` beyond 1 replica requires backing these with **Redis** (caches, rate limits, `jti` replay, flags) and **PostgreSQL / immutable object storage** (audit ledger, CQRS projections).
2. **Distributed Sparse Search:**
   - `BM25Okapi` is currently in-memory. At 10M–100M+ chunks, migrate sparse retrieval to **Milvus 2.4+ native Sparse/SPLADE vectors** or **OpenSearch** so dense + sparse hybrid search executes in a single distributed hop.
3. **Asymmetric / KMS-Backed JWT Signing & Live Vault Integration:**
   - Upgrade HS256 shared-secret JWTs to **RS256/EdDSA** with JWKS rotation and wire `SecretsProvider` directly to **HashiCorp Vault** or cloud KMS.
4. **Live Multi-Container Integration in CI:**
   - `scripts/docker_stack_smoke_test.py` validates the full 9-container Docker Compose stack, but CI currently runs unit/integration tests in-process without spinning up live Milvus, RabbitMQ, and Ollama containers.
5. **Real-Time Streaming Evaluation & Drift Alerts:**
   - Add automated background sampling of production queries into `RAGJudgeEvaluator` with Prometheus alerting on faithfulness or satisfaction drops.
