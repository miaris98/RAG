# Architectural Design Decisions, Tradeoffs & Measured Benchmarks (100+ Document Docker Stack Report)

> **Benchmark Environment:** Live 9-container Docker Compose stack (`fastapi-backend`, `frontend`, `milvus-standalone`, `milvus-etcd`, `milvus-minio`, `rabbitmq`, `ollama`, `rag-prometheus`, `rag-grafana`, `attu`) measured via `scripts/generate_100_docs_and_measure.py`.
> **Corpus Scale:** `103` documents (`10` generated multi-line PDFs with tables + `90` Markdown/Python runbooks + `3` seeded enterprise docs) -> `322` child chunks, `300` parent windows, `125` Knowledge Graph entities.

---

## 1. Executive Summary & Measured System Snapshot

Scaling the corpus from 3 sample documents to **103 documents (`322` chunks)** in Docker confirmed that rank-based hybrid fusion + Cross-Encoder reranking + Entity Graph (**Tri-Modal GraphRAG**) preserves **`1.000` MRR@5** and **`0.300` ContextPrecision@5** (highest across all 5 modes), while Semantic Caching cuts query latency by **30.8x** (`530.55 ms` cold -> `17.23 ms` warm). Even with Ollama offline (`llm_breaker_state=OPEN`), `L1_EXTRACTIVE_DEGRADED` kept `core_rag_available=true` with `0` rejected bulkhead calls.

| Subsystem / Operation | Measured Result (103-Doc Docker Run) | Key Operational Guarantee |
| :--- | :--- | :--- |
| **Markdown/Code Ingestion (`90` docs)** | `p50 = 163.18 ms`, `p95 = 272.03 ms`, `mean = 158.24 ms` | SHA-256 idempotent deduplication (`0` duplicates on re-ingest) |
| **PDF Ingestion (`10` PDFs w/ tables)** | `p50 = 177.88 ms`, `p95 = 10,280.49 ms` (cold model warmup) | Layout + Markdown table reconstruction + Late Chunking |
| **Cold RAG Query (`hybrid_graph_rerank`)** | `530.55 ms` | Pre-filtered RBAC + Tri-Modal RRF + Cross-Encoder + Citations |
| **Warm Semantic Cache Hit** | `17.23 ms` (**30.8x faster**) | Role- & tenant-isolated fast path (`cache_hit_verified=true`) |
| **CDC Upsert + Delete Roundtrip** | `120.48 ms` (`1` chunk superseded/removed) | Atomic version precedence + cache eviction |
| **Structured Taxonomy Classification** | `599.60 ms` (`payment_ledger_quorum_incident`, `73.2%` conf.) | Constrained enum output + verifiable rule quotes |
| **SQL OpsBot Read-Only Query** | `22.83 ms` (`4` live service rows) | Strict `SELECT`-only AST/keyword guard |
| **Zero Trust JWT (Issue + Verify + Replay)** | `33.79 ms` (`200` first, `401` replay) | `ttl <= 300s`, HS256, one-time `jti` replay cache |
| **GDPR Article 17 Surgical Erasure** | `47.05 ms` (`1` chunk deleted) | Erased across Dense, BM25, Graph, Cache, CQRS + SHA-256 chain |
| **Audit Chain & Prometheus** | `chain_valid = true` (`119` events), `28` metric series | Tamper-evident hash chain + 5s Prometheus scrape |

---

## 2. Chunking Strategies: Design Decisions, Tradeoffs & Measurements

**Files:** `backend/app/ingestion/chunking.py`, `backend/app/ingestion/pdf_parser.py`, `backend/app/retrieval/hybrid.py`

### Measured Comparison (30 Synthetic Docs per Strategy)

| Strategy | Docs | Chunks Created | Avg Chunks / Doc | Ingestion `p50` | Ingestion `p95` | Contextual Prefix | Late Chunking |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fixed`** | `30` | `30` | `1.0` | `75.29 ms` | `120.18 ms` | Off | Off |
| **`contextual_recursive`** | `30` | `120` | `4.0` | `163.18 ms` | `263.08 ms` | On (Anthropic prefix) | Off |
| **`parent_child`** | `30` | `120` | `4.0` | `210.61 ms` | `307.06 ms` | On (Anthropic prefix) | On (Perplexity blend) |

### Why & Tradeoffs per Strategy
1. **`fixed` Chunking:**
   - **Design:** Uniform character-window slicing (`~380` chars) without structural parsing.
   - **Why kept:** Useful baseline for ablation studies and lowest-latency ingestion (`p50 = 75.29 ms`, `2.8x` faster than `parent_child`).
   - **Tradeoff:** Splits Markdown tables and fenced code blocks mid-statement and loses section hierarchy, degrading retrieval precision on structured docs.
2. **`contextual_recursive` Chunking:**
   - **Design:** Splits along Markdown headings (`#`, `##`) while treating fenced code blocks (` ``` `) and Markdown tables (`| ... |`) as atomic units; extracts Python AST (`ast.parse`) and polyglot symbol names; prepends Anthropic-style `[Document | Section | Symbols]` contextual prefixes (optionally via Ollama).
   - **Why chosen:** Prevents code `#` comments from being misparsed as Markdown headers and preserves table columns together.
   - **Tradeoff:** `~2.2x` higher ingestion latency (`163.18 ms` p50) and `4x` more chunks than `fixed` on structured runbooks.
3. **`parent_child` (Small-to-Big) + Perplexity Late Chunking:**
   - **Design:** Indexes small `380`-char child chunks for sharp bi-encoder matching, maps each child to a `1200`-char parent window promoted before LLM synthesis, and blends document-level boundary embeddings into child vectors (`Late Chunking`).
   - **Why default:** Resolves the classic chunk-size dilemma—small chunks give high cosine precision without diluting multiple topics, while promoted parent windows preserve surrounding preconditions and remediation steps.
   - **Tradeoff:** Highest ingestion cost (`p50 = 210.61 ms`) and dual storage of child + parent text, justified by superior Top-1 grounding.
4. **NVIDIA-Style PDF Layout & Table Parser (`POST /ingest/pdf`):**
   - **Design:** Extracts PDF text via `pypdf` and reconstructs multi-column/whitespace-aligned tables into Markdown pipe tables before chunking (`p50 = 177.88 ms` warm).
   - **Tradeoff:** Lightweight pure-Python `pypdf` avoids multi-GB OCR/vision container bloat, at the cost of requiring text-selectable PDFs (scanned image-only PDFs would need OCR).

---

## 3. Retrieval Strategies & Tri-Modal GraphRAG: Design & Measurements

**Files:** `backend/app/retrieval/hybrid.py`, `backend/app/retrieval/entity_graph.py`, `backend/app/retrieval/reranker.py`, `backend/app/retrieval/postprocessor.py`, `backend/app/retrieval/semantic_ids.py`

### Measured 5-Mode Ablation on the 103-Document Corpus (`Top-K = 5`, 6 Golden Queries)

| Mode | HitRate@5 | MRR@5 | NDCG@5 | ContextPrecision@5 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`dense`** (Bi-Encoder `all-MiniLM-L6-v2`) | `1.000` | `0.7917` | `0.8348` | `0.2667` | `38.48 ms` |
| **`sparse`** (`BM25Okapi`) | `1.000` | `0.8333` | `0.8770` | `0.2500` | `1.29 ms` |
| **`hybrid_rrf`** (Dense + BM25, `k=60`) | `1.000` | `0.8333` | `0.8770` | `0.2333` | `32.97 ms` |
| **`hybrid_rerank`** (Hybrid RRF + Cross-Encoder) | `1.000` | **`1.0000`** | **`1.0000`** | `0.2667` | `789.73 ms` |
| **`hybrid_graph_rerank`** (**Tri-Modal GraphRAG**) | `1.000` | **`1.0000`** | `0.9866` | **`0.3000`** | `427.74 ms` |

### Why & Tradeoffs
1. **Pre-Filtered RBAC + Tenant Partition Sharding + Semantic IDs:**
   - **Why:** Filtering *after* top-K vector search causes **recall starvation** (returning 0 results when top matches belong to another tenant) and leaks existence. Pre-filtering `authorized_ids` + Airbnb tenant shards before scoring guarantees zero cross-tenant leakage.
   - **Tradeoff:** Requires metadata (`tenant_id`, `access_roles`) to be accurate at ingestion time.
2. **Dense vs. Sparse (BM25) vs. RRF (`k=60`):**
   - **Why:** Bi-encoders compress text into 384 floats and lose exact out-of-vocabulary IDs (`ERR_LEDGER_9042`), whereas BM25 misses semantic paraphrases. Rank-based RRF fuses both without brittle score normalization across bounded cosine and unbounded BM25 scores.
3. **Why Tri-Modal GraphRAG (`hybrid_graph_rerank`):**
   - **Design:** Adds a 1-hop Entity Knowledge Graph (`125` entities across `322` chunks in this run) alongside Dense and BM25, then reranks top RRF candidates with `ms-marco-MiniLM-L-6-v2`, applies **MMR** diversity, and **Lost-in-the-Middle** U-shaped prompt ordering.
   - **Measured Result:** Achieved the highest **`ContextPrecision@5` (`0.3000`)** and **`1.000` MRR@5**, promoting the exact answer to Rank 1 across every benchmark query.
   - **Tradeoff:** Cross-Encoder self-attention adds ~`400–790 ms` vs. `1.29 ms` BM25-only—acceptable for high-stakes enterprise RAG where Rank-1 accuracy prevents hallucinations.

---

## 4. Infrastructure Stack: Milvus, MinIO, Etcd, RabbitMQ, Prometheus & Grafana

| Component | Why Included | Architectural Role | Tradeoffs & Operational Notes |
| :--- | :--- | :--- | :--- |
| **Milvus (`:19530`)** + **Attu (`:9090`)** | Production ANN vector DB supporting scalar metadata filtering | Persists dense embeddings + tenant/role partitions; Attu provides collection inspection | Heavier memory footprint; mitigated by automatic local hybrid fallback (`L2_LOCAL_HYBRID_FALLBACK`) when Milvus is offline or buffering |
| **Etcd (`:2379`)** | Required Milvus coordination plane | Stores Milvus collection schemas, segment metadata, and cluster lease state | Single-node in Compose for local dev; production requires a 3-node quorum |
| **MinIO (`:9000`/`9001`)** | S3-compatible object storage for Milvus | Stores Milvus WAL/flushed vector segments and index files in `a-bucket` | Small inserts stay in Milvus's memory buffer until segment flush; requires `cgr.dev/chainguard/minio:latest` with `user: "0:0"` in Compose |
| **RabbitMQ (`:5672`/`15672`)** | Decouples slow ingestion/LLM jobs from HTTP request threads | Buffers `rag_jobs` and `rag_ingestion_jobs` with Management UI telemetry | Falls back transparently to local async threads when unreachable |
| **Prometheus (`:9091`)** | Time-series metrics scraper | Scrapes `fastapi-backend:8000/metrics/prometheus` every 5s (`28` active metric lines) | Pull-based scraping requires reachable `/metrics/prometheus` endpoint |
| **Grafana (`:3000`)** | Pre-provisioned visual observability | Renders **8 panels** (Latency `p50`/`p95`/`p99`, Queries/Cache/Guardrails/GDPR, Corpus size, Audit Chain status, Circuit Breakers, User Feedback, Prefix Cache, P4 Bulkheads) | Pre-provisioned JSON (`rag_observability.json`) ensures zero manual dashboard setup |

---

## 5. Security, Compliance & Data Governance Components

### A. PII & Prompt Injection Guardrails (`backend/app/guardrails/security.py`)
- **Why:** OWASP LLM01 (Prompt Injection) and LLM02 (Sensitive Information Disclosure) require scrubbing emails, SSNs, and API keys *before* indexing, auditing, or returning text to users.
- **Tradeoff:** Regex + signature rules run in `<1 ms` with zero false-negative risk on structured patterns, though nuanced semantic jailbreaks would benefit from an auxiliary classifier model in high-security deployments.

### B. Zero Trust Short-Lived Service JWTs & Secrets Hygiene (`backend/app/security/zero_trust.py`)
- **Design:** Pure-stdlib HS256 JWTs (`ttl_sec <= 300s`, `iss`, `aud`, `roles`, `jti`), one-time `jti` replay cache, per-service secret allow-lists, and automatic `redact_secrets()` inside `TamperEvidentAuditLogger`.
- **Measured Result:** Token issuance + `200` verification + `401` replay rejection completed in **`33.79 ms`** roundtrip.
- **Tradeoff:** Pure-stdlib HS256 avoids heavy crypto/Vault dependencies for local runs; production multi-replica deployments should swap to asymmetric RS256/EdDSA + external Vault/Redis `jti` store.

### C. GDPR Article 17 & Tamper-Evident SHA-256 Audit Ledger (`backend/app/guardrails/audit.py`)
- **Design:** `DELETE /documents/{doc_id}` surgically removes vectors, BM25 tokens, Knowledge Graph edges, Semantic Cache entries, and CQRS projections, then appends a hash-chained `gdpr_article17_erasure` event (`prev_hash -> event_hash`).
- **Measured Result:** Full erasure completed in **`47.05 ms`** (`1` chunk deleted), with `chain_valid = true` verified across **`119` audit events**.

### D. Incremental Change Data Capture — CDC (`POST /ingest/cdc`)
- **Design:** Keyed by `external_id`, atomically supersedes prior versions, evicts stale Semantic Cache entries, updates CQRS, and logs `cdc_upsert` / `cdc_delete`.
- **Measured Result:** Full upsert + delete roundtrip completed in **`120.48 ms`**.
- **Why:** Avoids expensive full-corpus re-indexing when source docs change in Notion/Confluence/Shopify.

---

## 6. Agentic & Specialized Query Components

### A. Structured Taxonomy Classification (`POST /query-structured`)
- **Design:** Combines DoorDash Inverted RAG (constraining output strictly to caller-supplied `allowed_labels`) with Ramp Rule-Grounded JSON (extracting verifiable supporting rules and `[Doc i]` citations).
- **Measured Result:** Classified `ERR_LEDGER_9042` as `payment_ledger_quorum_incident` (`73.2%` confidence) in **`599.60 ms`** under RBAC isolation.
- **Why:** Prevents LLMs from inventing arbitrary category labels in downstream routing/ticketing pipelines.

### B. Live SQL OpsBot (`backend/app/agent/sql_tool.py`)
- **Design:** Read-only SQLite interface (`service_inventory`, `active_incidents`) blocking all DML/DDL mutations (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `PRAGMA`, `;`) and fusing live operational metrics with static RAG runbooks.
- **Measured Result:** Returned `4` live service rows in **`22.83 ms`**.
- **Tradeoff:** SQLite keeps the demo self-contained; production would point the same read-only interface at a read-replica PostgreSQL/ClickHouse instance.

---

## 7. How Observability & Reliability Are Enforced Project-Wide

1. **Bounded Failure Isolation (`backend/app/resilience/`):** Every external dependency (`milvus_vector`, `ollama_llm`, `sql_opsbot`, `web_search`) is wrapped in a `BulkheadPool`, bounded `retry_with_backoff` with jitter, and `CircuitBreaker` state machines. In this 103-doc run, `milvus_vector` handled `32` bulkhead acquisitions with `0` rejections, and `L1_EXTRACTIVE_DEGRADED` kept `core_rag_available = true` when Ollama was unavailable.
2. **Decentralized Data Integrity (`backend/app/data/`):** `SagaOrchestrator` executes reverse compensating rollbacks on multi-store failures, while `CQRSReadProjection` serves `104` docs / `323` chunks (`version=107`) in $O(1)$ without locking write stores.
3. **End-to-End Distributed Tracing (`backend/app/observability/tracing.py`):** Every query and multi-agent supervisor run emits Langfuse-compatible parent-child spans (`query_routing -> rbac_prefilter -> dense/bm25/graph -> rrf -> cross_encoder -> crag -> generation -> self_rag`) rendered as ASCII waterfalls in Gradio Tab 2 & Tab 4.
4. **Progressive Delivery & Gated Chaos (`POST /ops/feature-flags`, `POST /ops/chaos`):** Deterministic SHA-256 Canary percentage rollouts (`0–100%`), Blue-Green `X-Release-Track` headers, and gated fault injection verify resilience before production outages.
5. **CI & Automated Quality Gates (`.github/workflows/edd.yml`):** Every push runs `92` TDD/security/spec-kit tests, frontend syntax checks, and the 5-mode IR regression gate (`HitRate@3 >= 0.90`, `MRR@3 >= 0.90`).

---

## 8. Reproducing This Benchmark Locally

```powershell
docker-compose up -d
$env:BACKEND_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe scripts\generate_100_docs_and_measure.py
```
