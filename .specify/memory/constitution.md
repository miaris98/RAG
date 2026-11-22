# Engineering Constitution — Staff+ Production RAG Microservices Platform

> **Spec-Kit Memory Anchor (`.specify/memory/constitution.md`)**
> Every feature, refactor, pull request, and operational change in this repository MUST uphold the nine non-negotiable architectural principles below. Violations are enforced automatically by `pytest backend/tests/ -v` and `.github/workflows/edd.yml`.

---

## Article I — Pre-Filtered Multi-Tenant RBAC (Zero Recall Starvation, Zero Leakage)
1. Authorization (`tenant_id` + `access_roles`) MUST execute **before** dense cosine scoring, BM25 scoring, and Entity Knowledge Graph expansion inside `backend/app/retrieval/hybrid.py`.
2. Post-retrieval filtering is strictly prohibited because it risks both unauthorized data exposure and empty top-$K$ result sets (recall starvation).
3. Semantic Cache keys MUST incorporate `(tenant_id, sorted(user_roles), retrieval_mode)` so cached responses can never leak across tenants or privilege boundaries.
4. Peer agent requests arriving via Google A2A (`POST /a2a`) MUST enforce **Anti-Confused-Deputy role ceilings** (`PEER_AGENT_ROLE_CEILINGS`), intersecting requested roles with the peer's maximum permitted role ceiling.

## Article II — Groundedness, Citation Traceability & Calibrated Abstention
1. Every internal answer MUST cite retrieved chunks via `[Doc i]` inline badges backed by structured `CitationItem` evidence quotes.
2. Raw Cross-Encoder logits MUST be calibrated to `[0, 1]` probabilities and gated by the Corrective RAG (CRAG) threshold (`RELEVANCE_THRESHOLD=0.25`) so low-confidence matches trigger abstention or controlled web fallback rather than hallucination.
3. The `rrf_breakdown` dictionary on `RetrievedChunk` MUST maintain the exact keys `{"dense", "sparse", "graph"}`; ranking multipliers (`freshness_boost`, `version_boost`) belong on dedicated typed fields.

## Article III — Idempotent, Structure-Aware Ingestion
1. All child chunks MUST use deterministic SHA-256 content-addressable IDs (`chk_<hash>`) so re-ingestion is 100% idempotent with zero duplicate vector poisoning.
2. Chunking MUST preserve Markdown tables, fenced code blocks (never mistaking `#` comments inside code fences for Markdown headers), Python AST / polyglot symbol names, and parent-child context windows.
3. Change Data Capture (`POST /ingest/cdc`) MUST supersede prior document versions cleanly and evict stale semantic cache entries.

## Article IV — Zero Trust, Secret Hygiene & Tamper-Evident Auditing
1. Internal service-to-service tokens (`backend/app/security/zero_trust.py`) MUST be short-lived (`ttl_sec <= 300s`), signed (HS256), audience-checked, and protected against replay via a one-time `jti` cache.
2. When `X-Service-Token` is present on `POST /a2a` or `POST /security/verify`, invalid, expired, or replayed tokens MUST return HTTP `401`.
3. `TamperEvidentAuditLogger` (`backend/app/guardrails/audit.py`) MUST sanitize PII, block prompt-injection artifacts, automatically scrub managed secrets via `secrets_provider.redact_secrets()`, and chain every event with SHA-256 (`prev_hash -> event_hash`).
4. All documentation, prompts, and MCP tool descriptions MUST remain free of OWASP LLM01/LLM02 and MCP tool-poisoning signatures (verified by `test_security_owasp_and_mcp.py` and `test_docs_spec_kit_completeness.py`).

## Article V — GDPR Article 17 Right to Be Forgotten
1. `DELETE /documents/{doc_id}` MUST surgically remove the target document across Dense vectors, BM25 index, Entity Knowledge Graph, Semantic Cache, and CQRS Read Projections, and append an immutable `gdpr_article17_erasure` event to the SHA-256 audit chain.

## Article VI — Resilient Microservices & Graceful Degradation
1. External/downstream calls (`ollama_llm`, `web_search`, `milvus_vector`, `sql_opsbot`) MUST be isolated behind per-dependency `BulkheadPool` compartments, bounded `retry_with_backoff` with jitter, and `CircuitBreaker` state machines.
2. When Ollama or external search fails (including under gated Chaos Engineering faults), the platform MUST degrade cleanly across `L0_FULL_OPERATIONAL -> L1_EXTRACTIVE_DEGRADED -> L2_LOCAL_HYBRID_FALLBACK -> L3_CACHE_ONLY_SURVIVAL` while preserving core citation-grounded RAG availability.

## Article VII — Decentralized Data Ownership, Sagas & CQRS
1. Each microservice boundary (`retrieval_service`, `sql_opsbot_service`, `compliance_audit_service`, `feedback_service`) owns its datastore via API contracts only (`shared_db_forbidden: True`).
2. Multi-step ingestion and GDPR erasure workflows MUST support Saga compensating rollbacks on downstream failure (`backend/app/data/saga.py`).
3. Write commands update a lock-free materialized `CQRSReadProjection` (`GET /cqrs/read-model`) decoupled from read queries.

## Article VIII — Least-Privilege Multi-Agent & Tool Isolation
1. Specialist Subagents (`retrieval_specialist`, `sql_ops_specialist`, `compliance_auditor`, `web_researcher`) MUST each hold only their minimum required tool scopes and execute under supervisor tracing.
2. Operational SQL access (`backend/app/agent/sql_tool.py`) MUST strictly allow read-only `SELECT` queries and reject all DML/DDL mutations.

## Article IX — Eval-Driven & Test-Driven Development (EDD + TDD)
1. Every capability MUST be introduced via TDD (Red -> Green -> Live Verification -> Merge) with zero regressions across the full `pytest backend/tests/ -v` suite.
2. CI (`.github/workflows/edd.yml`) enforces both full test passage and the 5-mode IR Ablation Gate (`HitRate@3 >= 0.90`, `MRR@3 >= 0.90` for `hybrid_graph_rerank`), plus zero documentation drift (`test_docs_spec_kit_completeness.py`).
