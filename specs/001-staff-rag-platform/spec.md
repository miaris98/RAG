# System Specification — Staff+ Production RAG Microservices Platform

> **Spec-Kit Artifact:** `specs/001-staff-rag-platform/spec.md`
> **Status:** Implemented & Verified (`89` automated TDD/security/spec tests, `50/50` industry RAG patterns, `P0–P4.6` complete)

---

## 1. Purpose & Problem Statement

Prototype RAG pipelines fail in enterprise production due to six systemic weaknesses: uncalibrated top-$K$ vector hallucinations, blind spots on exact identifiers/error codes, duplicate chunk poisoning, lack of pre-filtered multi-tenant RBAC, opaque black-box ReAct loops, and fragile single-point-of-failure downstream calls.

This platform provides a **100% local-capable, modular Production RAG Microservices Architecture** combining Tri-Modal Hybrid Retrieval, cyclic Corrective/Self-RAG orchestration, Parallel Specialist Subagents, MCP + Google A2A interoperability, GDPR/SOC2 compliance auditing, quantitative IR + LLM-as-a-Judge evaluation, and resilient microservices operations.

---

## 2. Target Personas & Consumers

| Persona / Consumer | Primary Interface | Goals |
| :--- | :--- | :--- |
| **Platform / Applied AI Engineer** | Gradio Studio (`:7860`), FastAPI Swagger (`:8000/docs`), CLI (`backend/debug.py`) | Ingest multi-format docs (Markdown, Code, PDF, CDC), inspect retrieval funnels & span waterfalls, benchmark 5 retrieval modes |
| **SRE / On-Call Incident Responder** | Tab 2 Query Studio, Agoda SQL OpsBot, Tab 4 Microservices Ops | Troubleshoot live incidents combining runbook RAG + read-only live SQL inventory, inspect degradation tiers (`L0–L3`), run Canary/Blue-Green & gated Chaos tests |
| **Security & Compliance Auditor** | Tab 4 Compliance Center, `/compliance/audit-report`, `/audit-logs`, `DELETE /documents/{doc_id}` | Verify SHA-256 audit chain integrity, inspect tenant/RBAC lineage, execute GDPR Article 17 erasures, confirm zero PII/secret leakage |
| **External AI Agents / IDEs** | MCP Server (`backend/mcp_server.py`), Google A2A (`/.well-known/agent.json`, `POST /a2a`) | Discover capabilities, invoke RBAC-scoped retrieval/SQL/taxonomy tools, exchange JSON-RPC 2.0 / SSE tasks under Zero Trust JWT & anti-confused-deputy ceilings |

---

## 3. Core User Journeys

### Journey A — Idempotent Multi-Modal Document Ingestion & CDC
1. User ingests Markdown, source code, or PDFs via `/ingest`, `/ingest/pdf`, `/ingest/cdc`, `/ingest-async`, or `/saga/ingest`.
2. Guardrails redact PII and flag prompt-injection payloads; Chunking Engine preserves tables/code fences, extracts Python AST/polyglot symbols, attaches Anthropic contextual prefixes, and optionally blends Perplexity late-chunking boundary embeddings.
3. Each chunk receives a deterministic SHA-256 ID, a Spotify 4-level `semantic_id`, and an Airbnb tenant `shard_id`; duplicates are skipped idempotently, stale caches are invalidated, and the CQRS read model & SHA-256 audit ledger update atomically.

### Journey B — Pre-Filtered RBAC Tri-Modal Query & Cyclic Self-Correction
1. Caller submits a query via `POST /query-stream` (SSE), `POST /submit-job` (RabbitMQ async), or `POST /agent/orchestrate` (multi-agent supervisor).
2. Adaptive Query Router classifies the query (`lexical_lookup`, `conceptual_hyde`, `multi_hop`, `standard`) and resolves multi-turn pronoun coreferences.
3. Hybrid Retrieval pre-filters by `(tenant_id, user_roles, shard_id)` before running Milvus/Dense (guarded by `milvus_vector` bulkhead), BM25, and 1-hop Entity Knowledge Graph -> Reciprocal Rank Fusion (`k=60`) with freshness decay & version boost -> Parent window promotion -> Cross-Encoder probability calibration -> MMR diversity -> Lost-in-the-Middle U-shaped reordering.
4. CRAG gates relevance; Self-RAG audits grounding and `[Doc i]` inline citations with supporting evidence quotes; if Ollama fails or chaos triggers, synthesis degrades cleanly to deterministic grounded extractive output.

### Journey C — Structured Taxonomy Classification & Live SQL OpsBot
1. Callers classify support/incident queries into caller-supplied taxonomy enums via `POST /query-structured` (DoorDash Inverted RAG + Ramp rule-grounded JSON).
2. Live operational questions automatically fuse Agoda OpsBot read-only SQLite service/incident telemetry (`/inventory/sql-status`, `/inventory/sql-query`) with internal RAG documents.

### Journey D — Quantitative Evaluation, Judge Audit & Observability
1. Engineers run `POST /evaluate` (or Tab 3) to execute the 5-mode IR ablation benchmark (`HitRate@K`, `MRR@K`, `NDCG@K`, `ContextPrecision@K`) plus `RAGJudgeEvaluator` (Faithfulness, Relevance, Dropbox filename/citation hallucination check, and 4-case human calibration).
2. Operators monitor Tab 4, `/traces` (Langfuse-compatible hierarchical span waterfalls), `/metrics/prometheus`, and Grafana's 8-panel dashboard (`:3000`).

### Journey E — Microservices Resilience, Zero Trust & Progressive Delivery
1. Internal services register heartbeats (`/discovery/register`), pass through `APIGatewayMiddleware` (`429 Retry-After`, `X-Request-Id`, `X-Release-Track`, security headers), and authenticate peer calls via short-lived HS256 JWTs (`POST /security/token`, `X-Service-Token` with `jti` replay protection).
2. SREs shift Canary rollout percentages and Blue-Green tracks (`POST /ops/feature-flags`) and validate fault tolerance with gated fault injection (`POST /ops/chaos`).

---

## 4. Functional Requirements Catalog

| ID | Domain | Requirement Summary | Primary Verification |
| :--- | :--- | :--- | :--- |
| `FR-001` | Ingestion | Table-, fenced-code-, and AST/polyglot-aware Parent-Child chunking with SHA-256 dedup | `test_rag_pipeline.py`, `test_todo_code_and_contextual_chunking.py` |
| `FR-002` | Ingestion | Anthropic contextual prefix enrichment (deterministic + optional Ollama LLM contextualizer) | `test_todo_code_and_contextual_chunking.py`, `test_todo_p3_1_ci_and_llm_contextualizer.py` |
| `FR-003` | Ingestion | Perplexity Late Chunking doc-boundary vector blending & NVIDIA layout/table PDF parser | `test_todo_late_chunking_and_pdf_parser.py` |
| `FR-004` | Ingestion | Shopify/Notion CDC incremental upsert/delete & RabbitMQ async ingestion queue | `test_todo_cdc_freshness_and_feedback.py`, `test_principal_rag_and_compliance.py` |
| `FR-005` | Retrieval | Pre-Filtered Multi-Tenant RBAC + Airbnb tenant shard routing + Spotify 4-level Semantic IDs | `test_staff_plus_features.py`, `test_todo_semantic_ids_and_sharding.py` |
| `FR-006` | Retrieval | Tri-Modal Hybrid Retrieval (Dense + BM25 + Entity Graph) via RRF (`k=60`) + freshness/version boosts | `test_staff_plus_features.py`, `test_todo_cdc_freshness_and_feedback.py` |
| `FR-007` | Retrieval | Cross-Encoder probability calibration, MMR diversity, negation awareness & Lost-in-the-Middle ordering | `test_principal_rag_and_compliance.py`, `test_tracing_visibility_and_citations.py` |
| `FR-008` | Caching | Role-scoped Semantic Cache, Labeled Query Sidecar, LinkedIn Shared-Prefix KV Cache & Instacart Pre-Warm | `test_staff_plus_features.py`, `test_todo_prefix_cache_and_prewarm.py` |
| `FR-009` | Agent | Cyclic LangGraph StateGraph (Adaptive Router -> CRAG -> Web -> Synthesis -> Self-RAG Retry Loop) | `test_principal_rag_and_compliance.py`, `test_staff_plus_features.py` |
| `FR-010` | Agent | Multi-turn conversational memory with pronoun coreference resolution | `test_principal_rag_and_compliance.py` |
| `FR-011` | Agent | DoorDash Inverted RAG constrained taxonomy & Ramp rule-grounded structured JSON (`POST /query-structured`) | `test_todo_inverted_rag_and_structured.py` |
| `FR-012` | Agent | Agoda OpsBot strictly read-only SQLite live inventory fused into RAG synthesis | `test_todo_agoda_sql_inventory_tool.py` |
| `FR-013` | Multi-Agent | 4 least-privilege Specialist Subagents orchestrated in parallel with supervisor span waterfall | `test_todo_p3_3_specialist_subagents.py` |
| `FR-014` | Protocols | RBAC-scoped MCP server (5 tools) & Google A2A server (Agent Card + JSON-RPC 2.0 / SSE + role ceilings) | `test_todo_p3_2_ui_mcp_and_grafana.py`, `test_todo_p3_4_google_a2a_protocol.py` |
| `FR-015` | Security | OWASP LLM01/LLM02 PII & Prompt Injection Guardrails + MCP/Doc anti-poisoning red-team tests | `test_security_owasp_and_mcp.py`, `test_staff_plus_features.py` |
| `FR-016` | Security | Zero Trust short-lived HS256 JWTs (`exp <= 300s`, `jti` replay cache) + Least-Privilege Secrets & auto-redaction | `test_todo_p4_4_zero_trust_and_secrets.py`, `test_todo_p4_6_runtime_wiring_and_ops_ui.py` |
| `FR-017` | Compliance | Tamper-evident SHA-256 hash-chained audit ledger, Compliance Dossier & GDPR Article 17 surgical erasure | `test_principal_rag_and_compliance.py`, `test_observability_and_audit_page.py` |
| `FR-018` | Evaluation | 5-mode IR Ablation (`HitRate@K`, `MRR@K`, `NDCG@K`, `ContextPrecision@K`) + LLM-as-a-Judge & Human Calibration | `test_rag_pipeline.py`, `test_todo_llm_judge_and_calibration.py` |
| `FR-019` | Observability | Langfuse-compatible hierarchical span waterfalls, Zendesk feedback loop, Prometheus & 8-panel Grafana dashboard | `test_observability_and_audit_page.py`, `test_release_polish_and_docker_stack.py` |
| `FR-020` | Resilience | Exponential backoff + jitter retries, per-dependency Bulkheads, Circuit Breakers & L0–L3 Degradation Matrix | `test_todo_p4_1_resilience_patterns.py`, `test_todo_p4_6_runtime_wiring_and_ops_ui.py` |
| `FR-021` | Traffic | Dynamic Service Discovery (TTL heartbeats/eviction), API Gateway Rate Limiter (`429`) & Istio STRICT mTLS mesh | `test_todo_p4_2_discovery_gateway_mesh.py` |
| `FR-022` | Data | Database-per-Service ownership boundaries, Ingestion/GDPR Sagas with compensating rollbacks & CQRS read model | `test_todo_p4_3_saga_and_cqrs.py` |
| `FR-023` | Delivery | Immutable K8s & Terraform IaC manifests, Canary `%` Feature Flags, Blue-Green `X-Release-Track` & Gated Chaos | `test_todo_p4_5_iac_progressive_delivery_chaos.py` |
| `FR-024` | Frontend | 4-Tab Gradio Observability, Eval, Compliance & P4 Microservices Ops Studio | `test_todo_p4_6_runtime_wiring_and_ops_ui.py` + live Playwright Chrome checks |
| `FR-025` | Docs / CI | EDD GitHub Actions gate, live Docker stack smoke runner & zero-drift Spec-Kit integrity tests | `test_release_polish_and_docker_stack.py`, `test_docs_spec_kit_completeness.py` |

---

## 5. Non-Functional SLOs & Quality Gates

- **Retrieval Quality Gate (CI-enforced):** `hybrid_graph_rerank` MUST achieve `HitRate@3 >= 0.90` and `MRR@3 >= 0.90` on the Golden QA benchmark.
- **Cache Fast-Path Latency:** Semantic Cache exact hits return in `< 5 ms` without invoking retrieval or LLM tiers.
- **Availability Under Faults:** When Ollama or DuckDuckGo trips its Circuit Breaker or encounters Chaos injection, `core_rag_available` remains `True` via deterministic citation-grounded extractive synthesis.
- **Audit Integrity:** `audit_logger.verify_chain()` MUST evaluate `True` across all retained events; any single-byte post-hoc modification invalidates the chain.
- **Zero Documentation Drift:** `test_docs_spec_kit_completeness.py` fails CI if any FastAPI route, MCP tool, A2A skill, or `backend/app/*` package is undocumented.
