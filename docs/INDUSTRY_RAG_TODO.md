# 50-Company Production RAG Tech Blog Study & Project TODO

> Synthesized from engineering blogs, architecture whitepapers, and production post-mortems across **50 technology companies**—mapped directly to concrete TODOs for this repository (`backend/app/` & `frontend/app.py`).

---

## 1. 50-Company Engineering Blog Matrix & Gap Analysis

### A. Big Tech & Foundation AI Labs (10 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 1 | **Google** | Structure-aware chunking, content-hash incremental indexing, distributed LLMOps tracing | Implemented (SHA-256 + hierarchical span waterfall) |
| 2 | **Netflix** | Stateful ingestion orchestration, avoiding reindex tax, distributed observability | Implemented (RabbitMQ async ingest + Prometheus/Grafana) |
| 3 | **Meta** | Dense + sparse retrieval scaling, avoiding unnecessary agentic complexity | Implemented |
| 4 | **Microsoft** | Retrieval-time ACLs inside vector index; treat retrieved text as untrusted input | Implemented (Pre-filtered RBAC + Guardrails) |
| 5 | **AWS** | OpenSearch Hybrid BM25 + Vector, Step Functions async ingestion, X-Ray tracing | Implemented (`POST /ingest-async` + distributed traces) |
| 6 | **Anthropic** | Contextual Retrieval (contextual chunk prefix + Contextual BM25, -49% failures) | Implemented (`enable_contextual_enrichment` prefixing) |
| 7 | **OpenAI** | Two-stage bi-encoder retrieval + cross-encoder reranking; avoid context noise | Implemented |
| 8 | **Cohere** | Cross-Encoder stage-2 precision reranking over top-50/top-12 candidates | Implemented |
| 9 | **Perplexity** | Document-boundary-aware contextual embeddings (`pplx-embed-context-v1`) | Implemented (`enable_late_chunking` doc-vector blending) |
| 10 | **NVIDIA** | Multimodal table/chart extraction (NeMo Retriever) + Judge/Reward scoring | Implemented (`POST /ingest/pdf` layout/table parser + `judge.py`) |

### B. Social, Search & Media Platforms (8 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 11 | **Reddit** | Append-only mini-indices + periodic merges for high-frequency freshness; Hybrid BM25+HNSW | Implemented (`freshness_boost` time-decay + CDC upserts) |
| 12 | **LinkedIn** | Retrieval-as-Ranking (RAR), shared-prefix prompt caching/amortization, pruned SLMs | Implemented (`prefix_cache.py` KV prefix amortization + Prometheus) |
| 13 | **Pinterest** | *Manas* distributed hybrid semantic + lexical search platform | Implemented |
| 14 | **Spotify** | Semantic IDs & multi-task search + recommendation retrieval | Implemented (`semantic_ids.py` 4-level `sem.*` IDs + prefix filter) |
| 15 | **Slack** | Scoped context retrieval strictly inheriting workspace/channel ACLs | Implemented |
| 16 | **Yelp** | FAISS PQ sub-20ms @ p99 + reranking folding in business-logic & freshness scores | Implemented (`freshness_boost` & `version_boost` in RRF) |
| 17 | **Cursor** | AST-aware code chunking, symbol resolution, minimizing prompt pollution | Implemented (Atomic fenced code + Python AST + MMR) |
| 18 | **GitHub** | Hybrid code + documentation indexing with symbol-aware retrieval | Implemented (`extract_code_symbols` AST/polyglot metadata) |

### C. Marketplaces, Delivery & Travel (9 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 19 | **Uber** | *Genie* EAg-RAG: LangGraph multi-agent loop with table/metadata extraction & refinement | Implemented (Cyclic LangGraph Self-RAG + table chunker) |
| 20 | **Airbnb** | Contrastive Two-Tower EBR + sharded scatter-gather ANN serving | Implemented (Tenant partition shard routing + `/corpus/shards-and-clusters`) |
| 21 | **DoorDash** | "Inverted RAG": ANN retrieves top-100 taxonomy concepts to constrain LLM output space | Implemented (`POST /query-structured` constrained enum mode) |
| 22 | **Instacart** | Head/Tail traffic splitting: offline RAG cache for head queries vs. real-time LLM for tail | Implemented (`POST /cache/prewarm` + Semantic Query Cache) |
| 23 | **Lyft** | `lyft2vec` graph embeddings + *AI Assist* multi-stage support RAG with strict evals | Implemented (Tri-Modal Entity Graph + 5-Mode Ablations) |
| 24 | **Booking.com** | `Orchestrator -> Moderation -> Agent -> RAG` + 1-year Judge-LLM evaluation lessons | Implemented (`RAGJudgeEvaluator` + Guardrails + Router) |
| 25 | **Agoda** | DB-backed live inventory retrieval + *OpsBot* (OpenTelemetry traces + post-mortem RAG) | Implemented (`sql_tool.py` read-only SQLite OpsBot + `/inventory/sql-*`) |
| 26 | **Expedia Group** | MCP protocol tool bridge + Neo4j entity graph unifying fragmented data | Implemented (`mcp_server.py` RBAC MCP tools + Entity Graph) |
| 27 | **Grab** | *LLM-Kit* standardized FastAPI+LangGraph+OTel boilerplate + negation-aware vector search | Implemented (Negation-aware reranker + distributed trace spans) |

### D. SaaS, Collaboration & Developer Platforms (13 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 28 | **Notion** | Dual-path ingestion (Spark batch + Kafka/Debezium CDC streaming) + tenant shards | Implemented (`POST /ingest-async`, `POST /ingest/cdc`, tenant shards) |
| 29 | **Stripe** | *Kai* hybrid RAG + tool-calling with sub-200ms incremental index updates | Implemented (Hybrid RAG + SQL/MCP tools + CDC upserts) |
| 30 | **Shopify** | Change Data Capture (CDC) ingestion for semi-structured catalog schemas | Implemented (`POST /ingest/cdc` incremental upsert/delete) |
| 31 | **Dropbox** | *Dropbox Dash*: human-calibrated LLM-as-a-judge to catch hallucinated filenames | Implemented (`judge.py` filename/citation audit + human calibration) |
| 32 | **Figma** | Hierarchical metadata filtering (permissions, team spaces, version control status) | Implemented (Pre-filtered RBAC + `version_boost` + CDC superseding) |
| 33 | **Canva** | Multi-modal asset/design token retrieval with permission-aware metadata | Implemented (RBAC metadata + PDF/table/code extraction) |
| 34 | **Cloudflare** | Edge Queues async ingestion + FTS5 + Vectorize + deterministic fallback/error boundaries | Implemented (`CircuitBreaker` for Ollama/DDG + RabbitMQ fallback) |
| 35 | **Datadog** | Embedding latency tracking, dimensionality vs. cost, token & leakage telemetry | Implemented (`/metrics`, `/metrics/prometheus`, p50/p95/p99, prefix tokens) |
| 36 | **Atlassian** | Nested Confluence/Jira doc parsing, webhook freshness, strict ACL boundaries | Implemented (Multi-turn session memory + MD nesting + CDC freshness) |
| 37 | **Salesforce** | Relational schema + enriched indexing lifting retrieval accuracy | Implemented (SQL inventory tool + Contextual/AST enriched indexing) |
| 38 | **Zendesk** | Transitioning keyword search to bi-encoder semantic search trained on user clicks | Implemented (`POST/GET /feedback` thumbs-up/down + Prometheus KPIs) |
| 39 | **HubSpot** | Vector-as-a-Service (VaaS) over Qdrant, weighted reranking, automated ACLs | Implemented (Weighted Tri-Modal RRF + automated RBAC pre-filtering) |
| 40 | **Twilio** | Production LLM-as-a-Judge alignment against human expectations | Implemented (`run_human_calibration_check()` in `judge.py` & `benchmark.py`) |

### E. FinTech, Legal & Regulated Enterprise (10 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 41 | **Rippling** | **Eval-Driven Development (EDD)**: smoke evals on every commit, cost-uncertainty-lag | Implemented (`.github/workflows/edd.yml` + 50 TDD & 5-mode ablation tests) |
| 42 | **Bloomberg** | *EvidenT* provable source traceability + RAG Safety & Risk Taxonomies | Implemented (`CitationItem` exact quote extraction + OWASP guardrails) |
| 43 | **Capital One** | Platform-first governance + *FB-RAG* (Forward-Backward RAG for constraints) | Implemented (Cyclic Self-RAG retry loop + governance audit dossier) |
| 44 | **JPMorgan Chase** | Regulated financial compliance, strict data lineage & auditability | Implemented (Tamper-evident SHA-256 hash-chained audit ledger + GDPR Art. 17) |
| 45 | **Ramp** | Auditable RAG grounding classification in verifiable rules & data lineage | Implemented (`POST /query-structured` verifiable rules + audit hash) |
| 46 | **Thomson Reuters** | Legal two-stage bi-encoder + cross-encoder separating retrieval vs. generation evals | Implemented (IR ablation metrics + P2.1 Judge generation audit) |
| 47 | **Plaid** | Regulated fintech data security & compliance-constrained retrieval | Implemented (RBAC isolation + PII redaction + compliance audit page) |
| 48 | **Block** | *Labeled Queries* sidecar vector store (vetted human queries take priority over raw logs) | Implemented (`labeled_queries.py` vetted Golden QA sidecar) |
| 49 | **Square** | Cosine similarity over domain embeddings for item/intent matching | Implemented |
| 50 | **Monzo** | Lightweight precomputed vector distance serving for fast help-center retrieval | Implemented |

---

## 2. Prioritized Project TODO Roadmap (Mapped to Our Codebase)

### P0 — High-Impact Retrieval & Agent Reliability (Interview Differentiators)
- [x] **P0.1 Cyclic Self-Correction in LangGraph** (`backend/app/agent/graph.py`, `nodes.py`)
  - *Inspired by:* **Uber (Genie)**, **Capital One (FB-RAG)**
  - Implemented cyclic conditional edge after `verify_faithfulness_self_rag`: when `faithfulness_score < 0.75` and `retry_count < max_retries`, broadens `top_k`, rewrites the query using critique feedback, and loops back to `retrieve_and_rerank`.
- [x] **P0.2 Negation-Aware Reranking & MMR Diversity + Lost-in-the-Middle Reordering** (`backend/app/retrieval/postprocessor.py`)
  - *Inspired by:* **Grab**, **OpenAI**, **Cursor**
  - Implemented Maximal Marginal Relevance (MMR) deduplication, U-shaped Lost-in-the-Middle reordering (`[Rank 1, Rank 3, ..., Rank 2]`), and explicit negation penalty detection.
- [x] **P0.3 Table-Header & Code-Block-Aware Chunking + Optional Contextualizer** (`backend/app/ingestion/chunking.py`)
  - *Inspired by:* **Anthropic**, **NVIDIA**, **Cursor**, **GitHub**, **Atlassian**
  - Preserves Markdown table headers across split boundaries, keeps fenced code blocks atomic (ignoring `#` comments inside fences), extracts Python AST + polyglot code symbols (`code_symbols`, `code_languages`), and supports optional Anthropic-style contextual document prefixes (`enable_contextual_enrichment`).
- [x] **P0.4 Multi-Turn Conversational Memory & Coreference Resolution** (`backend/app/memory/conversation.py`)
  - *Inspired by:* **Booking.com**, **Lyft (AI Assist)**
  - Implemented `session_id` sliding-window memory resolving pronouns (`"it"`, `"its remediation"`, `"that error"`) into standalone queries before retrieval.

### P1 — Production Operations, Observability & Compliance
- [x] **P1.1 Circuit Breakers + `GET /metrics` & `/metrics/prometheus` (p50/p95/p99, Cache Hit Rate, Error Rate)** (`backend/app/observability/`)
  - *Inspired by:* **Cloudflare**, **Datadog**, **AWS**
  - Wrapped Ollama & DuckDuckGo calls with `CLOSED -> OPEN -> HALF_OPEN` circuit breakers and exposed JSON + Prometheus latency, guardrail, feedback, and corpus metrics.
- [x] **P1.2 GDPR Surgical Deletion (`DELETE /documents/{doc_id}`), Freshness Decay & CDC Upserts (`POST /ingest/cdc`)** (`backend/main.py`, `hybrid.py`, `pipeline.py`)
  - *Inspired by:* **Figma**, **Notion**, **Shopify**, **Reddit**, **Yelp**, **JPMorgan Chase**
  - Deletes a single `doc_id` across Dense, BM25, Entity Graph, and Semantic Cache, supports atomic `external_id` CDC upserts/deletes, and applies exponential freshness time-decay + version precedence boosts in RRF scoring.
- [x] **P1.3 Async Ingestion Queue (`rag_ingestion_jobs`)** (`backend/app/worker/consumer.py`)
  - *Inspired by:* **Cloudflare**, **Notion**, **AWS**
  - Offloads heavy ingestion to RabbitMQ (`POST /ingest-async`) with local thread fallback.
- [x] **P1.4 PII-Safe Tamper-Evident Audit Logging & User Feedback Loop** (`backend/app/guardrails/audit.py`, `backend/app/observability/feedback.py`)
  - *Inspired by:* **Bloomberg**, **Ramp**, **Zendesk**, **JPMorgan Chase**
  - Emits SHA-256 hash-chained, PII-redacted audit records linking responses to exact `doc_id`, `chunk_id`, and `user_roles`, plus `POST /feedback` & `GET /feedback` satisfaction tracking.

### P2 — Evaluation, Calibration & Ecosystem Integration
- [x] **P2.1 LLM-as-a-Judge & Human-Calibrated Evals in `benchmark.py` & `judge.py`**
  - *Inspired by:* **Booking.com**, **Dropbox**, **Twilio**, **NVIDIA**
  - Added Faithfulness, Answer Relevance, and Dropbox-style filename/citation hallucination verification with optional local Ollama Judge, deterministic rubric fallback, and 4-case Human-Calibration Agreement check in `POST /evaluate` and Gradio Tab 3.
- [x] **P2.2 GitHub Actions CI Eval-Driven Development (EDD) Gate** (`.github/workflows/edd.yml`)
  - *Inspired by:* **Rippling**
  - Runs `pytest` + the 5-mode ablation benchmark gate on every PR and fails if `MRR@3` or `HitRate@3` regresses.
- [x] **P2.3 MCP Server, Labeled-Query Sidecar & Inverted RAG Structured Classification** (`backend/mcp_server.py`, `labeled_queries.py`, `structured.py`)
  - *Inspired by:* **Expedia Group**, **Block**, **Instacart**, **DoorDash**, **Ramp**
  - Exposes RBAC-scoped MCP tools (`retrieve_hybrid`, `ingest_document`, `gdpr_erase_document`), a vetted Labeled Queries sidecar cache, and `POST /query-structured` constrained taxonomy / rule-grounded JSON output.

### P3 — Multi-Agent Subagents, A2A Protocol & Full-Stack Production Parity
- [x] **P3.1 CI/Docker Dependency Parity & Optional Ollama LLM Contextualizer** (`backend/requirements.txt`, `.github/workflows/edd.yml`, `backend/app/ingestion/chunking.py`)
  - Add `python-multipart` (and `requests` in CI) so `POST /ingest/pdf` boots cleanly in Docker Compose and GitHub Actions EDD runs.
  - Add optional `use_llm_contextualizer` flag in `ChunkingEngine` calling local Ollama (guarded by `CircuitBreaker`) with seamless deterministic extractive fallback.
- [x] **P3.2 Full Gradio UI, MCP & Grafana Observability Parity** (`frontend/app.py`, `backend/mcp_server.py`, `observability/grafana/`)
  - Add **PDF Upload (`POST /ingest/pdf`)** and **CDC Upsert/Delete (`POST /ingest/cdc`)** controls to Gradio Tab 1.
  - Surface **User Feedback KPIs**, **Airbnb Shards & Spotify Semantic Clusters**, **LinkedIn Prefix Cache Stats + Instacart Pre-Warm button**, and **Agoda OpsBot Read-Only SQL Inspector** in Gradio Tab 4.
  - Expose `query_live_sql_inventory` and `classify_structured_taxonomy` in `backend/mcp_server.py`, and chart `rag_user_feedback_total`, `rag_user_satisfaction_rate`, `rag_prefix_cache_hits_total`, and `rag_prefix_cache_amortized_tokens_total` in Grafana.
- [x] **P3.3 Specialist Subagents & Parallel Supervisor Orchestration** (`backend/app/agent/subagents.py`, `POST /agent/orchestrate`)
  - *Inspired by:* **Anthropic Multi-Agent Research**, **Uber (Genie)**, **LangGraph Supervisor**
  - Implement a **Supervisor Agent** that decomposes complex queries and delegates in parallel to isolated, least-privilege **Specialist Subagents**: `retrieval_specialist` (Tri-Modal Hybrid RAG), `sql_ops_specialist` (read-only live SQL inventory/incidents), `compliance_auditor` (guardrails + structured policy rules + P2.1 hallucination judge), and `web_researcher` (circuit-breaker external search).
  - Emit per-subagent child spans into `TraceCollector` for waterfall visualization and expose multi-agent delegation telemetry in Gradio Tab 2.
- [x] **P3.4 Google A2A (Agent-to-Agent) Protocol Server & Confused-Deputy Security** (`backend/app/a2a/`, `GET /.well-known/agent.json`, `POST /a2a`)
  - *Inspired by:* **Google A2A Protocol Specification**, **OWASP Agentic Top 10**
  - Publish a spec-compliant **A2A Agent Card** at `GET /.well-known/agent.json` advertising skills (`enterprise_rag_qa`, `live_sql_ops_audit`, `taxonomy_rule_classifier`, `multi_agent_research`), capabilities, and RBAC auth requirements.
  - Implement JSON-RPC 2.0 `POST /a2a` supporting `tasks/send`, `tasks/get`, `tasks/cancel`, and SSE streaming (`tasks/sendSubscribe`) with typed A2A `Artifact` outputs and `submitted -> working -> completed | failed | canceled` task states.
  - Enforce **A2A anti-confused-deputy RBAC delegation** (intersecting peer-agent allowed scopes with caller roles) and log every cross-agent A2A handoff to the SHA-256 tamper-evident audit ledger.

### P4 — Resilient Microservices Architecture (Fault Tolerance, Traffic, Data, Zero Trust & Chaos)
- [x] **P4.1 Resilience & Fault Tolerance (Retries + Jitter, Bulkheads, Graceful Degradation)** (`backend/app/resilience/`)
  - Add bounded **Retries with Exponential Backoff & Decorrelated Jitter** around Ollama, DuckDuckGo, Milvus, and RabbitMQ clients (paired with existing Circuit Breakers).
  - Add per-dependency **Bulkhead Isolation** (`BulkheadPool` concurrency semaphores / isolated pools for `milvus_vector`, `ollama_llm`, `sql_opsbot`, `web_search`) with Prometheus saturation gauges.
  - Formalize a **Graceful Degradation Matrix** (`L0 Full -> L1 Extractive/Cached -> L2 BM25+Graph -> L3 Cache-Only`) exposed at `GET /health/resilience`.
- [x] **P4.2 Service Discovery, API Gateway & Service Mesh Readiness** (`backend/app/gateway/`, `deploy/mesh/`)
  - Implement a **Dynamic Service Discovery Registry** with heartbeat TTL, health checks, and stale-instance eviction (`POST /discovery/register`, `GET /discovery/services`) compatible with Consul / Kubernetes DNS.
  - Add centralized **API Gateway Middleware** for per-tenant/per-IP token-bucket rate limiting (`429 Retry-After`), `X-Request-Id` propagation, and security headers.
  - Add **Service Mesh / mTLS** manifests (`deploy/mesh/` Istio `PeerAuthentication` `STRICT` mTLS + `DestinationRule` retry/outlier policies).
- [x] **P4.3 Decentralized Data Management: Database-per-Service, Saga Pattern & CQRS** (`backend/app/data/`)
  - Enforce **Database-per-Service** boundaries across Vector/Search, SQL OpsBot, Audit Ledger, and Feedback stores via strict service interfaces.
  - Implement **Saga Pattern with Compensating Transactions** (`IngestionSaga` rolling back partial Vector/BM25/Graph writes on step failure; `GDPRErasureSaga` coordinating multi-store erasure with compensation logs).
  - Implement **CQRS (Command Query Responsibility Segregation)** separating write commands (`/ingest`, `/ingest/cdc`, `DELETE /documents/{doc_id}`) from lock-free read-optimized materialized projections serving `/documents`, `/corpus/shards-and-clusters`, and `/compliance/audit-report`.
- [x] **P4.4 Zero Trust Security & Least-Privilege Secrets Hygiene** (`backend/app/security/zero_trust.py`)
  - Implement **Zero Trust Short-Lived Service JWTs** (`iss`, `sub`, `aud`, `roles`, `exp <= 300s`, `jti` replay cache) validated via `X-Service-Token` across internal service & A2A peer boundaries.
  - Add a **Least-Privilege Secrets Provider** abstraction (env + HashiCorp Vault/KMS adapter) with automated secret-leakage redaction in logs, traces, and audit events.
- [x] **P4.5 Immutable IaC, Progressive Delivery & Chaos Engineering** (`deploy/`, `backend/app/resilience/chaos.py`)
  - Add **Immutable Infrastructure & IaC** manifests (`deploy/k8s/` Deployments, Services, HPA, NetworkPolicies, PDBs + `deploy/terraform/` skeleton).
  - Implement **Progressive Delivery** via a runtime `FeatureFlagManager` (`GET/POST /ops/feature-flags`) supporting deterministic percentage **Canary** rollouts (`1%–100%`) and **Blue-Green** release track tagging (`X-Release-Track`).
  - Implement a gated **Chaos Engineering Fault Injector** (`POST /ops/chaos`) to inject controlled latency spikes or downstream faults (`ollama`, `duckduckgo`, `milvus`) and verify Circuit Breakers, Bulkheads, and Graceful Degradation in automated tests & live demos.
- [x] **P4.6 End-to-End Microservices Runtime Wiring & Ops UI Parity** (`nodes.py`, `hybrid.py`, `audit.py`, `main.py`, `frontend/app.py`)
  - Wire `bulkhead_registry`, `retry_with_backoff`, and `chaos_controller.maybe_inject` directly into live Ollama, DuckDuckGo, and Dense/Milvus execution paths.
  - Enforce Zero Trust `X-Service-Token` JWT + `jti` replay validation on `POST /a2a` whenever supplied by peer agents.
  - Automatically scrub managed secrets via `secrets_provider.redact_secrets()` inside `TamperEvidentAuditLogger.log_event()`.
  - Surface live Resilience, Service Discovery, CQRS, Canary Feature Flags, and Chaos Engineering controls inside Gradio Tab 4.
