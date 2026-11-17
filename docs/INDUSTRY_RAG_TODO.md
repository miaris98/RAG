# 🌐 50-Company Production RAG Tech Blog Study & Project TODO

> Synthesized from engineering blogs, architecture whitepapers, and production post-mortems across **50 technology companies**—mapped directly to concrete TODOs for this repository (`backend/app/` & `frontend/app.py`).

---

## 1. 50-Company Engineering Blog Matrix & Gap Analysis

### A. Big Tech & Foundation AI Labs (10 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 1 | **Google** | Structure-aware chunking, content-hash incremental indexing, distributed LLMOps tracing | ✅ Partial (SHA-256 + stage latencies) |
| 2 | **Netflix** | Stateful ingestion orchestration, avoiding reindex tax, distributed observability | ✅ Partial (RabbitMQ + latency trace) |
| 3 | **Meta** | Dense + sparse retrieval scaling, avoiding unnecessary agentic complexity | ✅ Implemented |
| 4 | **Microsoft** | Retrieval-time ACLs inside vector index; treat retrieved text as untrusted input | ✅ Implemented (Pre-filtered RBAC + Guardrails) |
| 5 | **AWS** | OpenSearch Hybrid BM25 + Vector, Step Functions async ingestion, X-Ray tracing | ✅ Partial (Missing async `/ingest` queue) |
| 6 | **Anthropic** | Contextual Retrieval (LLM-generated chunk prefix + Contextual BM25, -49% failures) | ✅ Partial (Template header; needs LLM contextualizer) |
| 7 | **OpenAI** | Two-stage bi-encoder retrieval + cross-encoder reranking; avoid context noise | ✅ Implemented |
| 8 | **Cohere** | Cross-Encoder stage-2 precision reranking over top-50/top-12 candidates | ✅ Implemented |
| 9 | **Perplexity** | Document-boundary-aware contextual embeddings (`pplx-embed-context-v1`) | 🔲 TODO (Late Chunking / Context-aware boundary pass) |
| 10 | **NVIDIA** | Multimodal table/chart extraction (NeMo Retriever) + Judge/Reward scoring | 🔲 TODO (Table-aware chunker & PDF layout parser) |

### B. Social, Search & Media Platforms (8 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 11 | **Reddit** | Append-only mini-indices + periodic merges for high-frequency freshness; Hybrid BM25+HNSW | 🔲 TODO (Freshness/time-decay boost + incremental merge) |
| 12 | **LinkedIn** | Retrieval-as-Ranking (RAR), shared-prefix prompt caching/amortization, pruned SLMs | 🔲 TODO (Shared-prefix prompt caching / KV reuse) |
| 13 | **Pinterest** | *Manas* distributed hybrid semantic + lexical search platform | ✅ Implemented |
| 14 | **Spotify** | Semantic IDs & multi-task search + recommendation retrieval | 🔲 TODO (Entity/Semantic ID clustering) |
| 15 | **Slack** | Scoped context retrieval strictly inheriting workspace/channel ACLs | ✅ Implemented |
| 16 | **Yelp** | FAISS PQ sub-20ms @ p99 + reranking folding in business-logic & freshness scores | 🔲 TODO (Recency/version decay in reranker) |
| 17 | **Cursor** | AST-aware code chunking, symbol resolution, minimizing prompt pollution | 🔲 TODO (AST/code-block-aware chunker + MMR) |
| 18 | **GitHub** | Hybrid code + documentation indexing with symbol-aware retrieval | 🔲 TODO (Code symbol metadata extraction) |

### C. Marketplaces, Delivery & Travel (9 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 19 | **Uber** | *Genie* EAg-RAG: LangGraph multi-agent loop with table/metadata extraction & refinement | ✅ Partial (Needs cyclic self-correction loop) |
| 20 | **Airbnb** | Contrastive Two-Tower EBR + sharded scatter-gather ANN serving | ✅ Partial (Needs Milvus partition/shard routing) |
| 21 | **DoorDash** | "Inverted RAG": ANN retrieves top-100 taxonomy concepts to constrain LLM output space | 🔲 TODO (Constrained taxonomy/enum output mode) |
| 22 | **Instacart** | Head/Tail traffic splitting: offline RAG cache for head queries vs. real-time LLM for tail | ✅ Partial (Semantic Cache; add head-query pre-warming) |
| 23 | **Lyft** | `lyft2vec` graph embeddings + *AI Assist* multi-stage support RAG with strict evals | ✅ Partial (Entity Graph + Eval) |
| 24 | **Booking.com** | `Orchestrator -> Moderation -> Agent -> RAG` + 1-year Judge-LLM evaluation lessons | 🔲 TODO (LLM-as-a-Judge evaluation mode) |
| 25 | **Agoda** | DB-backed live inventory retrieval + *OpsBot* (OpenTelemetry traces + post-mortem RAG) | 🔲 TODO (Structured SQL/API tool alongside vector RAG) |
| 26 | **Expedia Group** | MCP protocol tool bridge + Neo4j entity graph unifying fragmented data | 🔲 TODO (Expose RAG engine as MCP tool server) |
| 27 | **Grab** | *LLM-Kit* standardized FastAPI+LangGraph+OTel boilerplate + negation-aware vector search | 🔲 TODO (Negation-aware reranker & OpenTelemetry) |

### D. SaaS, Collaboration & Developer Platforms (13 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 28 | **Notion** | Dual-path ingestion (Spark batch + Kafka/Debezium CDC streaming) + tenant shards | 🔲 TODO (Async Kafka/RabbitMQ ingestion worker) |
| 29 | **Stripe** | *Kai* hybrid RAG + tool-calling with sub-200ms incremental index updates | ✅ Partial |
| 30 | **Shopify** | Change Data Capture (CDC) ingestion for semi-structured catalog schemas | 🔲 TODO (Webhook/CDC incremental upsert endpoint) |
| 31 | **Dropbox** | *Dropbox Dash*: human-calibrated LLM-as-a-judge to catch hallucinated filenames | 🔲 TODO (Golden Judge calibration + filename/citation check) |
| 32 | **Figma** | Hierarchical metadata filtering (permissions, team spaces, version control status) | 🔲 TODO (Document version precedence & superseding) |
| 33 | **Canva** | Multi-modal asset/design token retrieval with permission-aware metadata | ✅ Partial (RBAC metadata) |
| 34 | **Cloudflare** | Edge Queues async ingestion + FTS5 + Vectorize + deterministic fallback/error boundaries | 🔲 TODO (Circuit Breakers for Ollama/DuckDuckGo) |
| 35 | **Datadog** | Embedding latency tracking, dimensionality vs. cost, token & leakage telemetry | 🔲 TODO (`GET /metrics` Prometheus/p95/p99 endpoint) |
| 36 | **Atlassian** | Nested Confluence/Jira doc parsing, webhook freshness, strict ACL boundaries | 🔲 TODO (Multi-turn session memory & Jira/MD nesting) |
| 37 | **Salesforce** | Relational schema + enriched indexing lifting retrieval accuracy | ✅ Partial |
| 38 | **Zendesk** | Transitioning keyword search to bi-encoder semantic search trained on user clicks | 🔲 TODO (User feedback thumbs-up/down logging) |
| 39 | **HubSpot** | Vector-as-a-Service (VaaS) over Qdrant, weighted reranking, automated ACLs | ✅ Partial |
| 40 | **Twilio** | Production LLM-as-a-Judge alignment against human expectations | 🔲 TODO (LLM-as-a-Judge scoring in `benchmark.py`) |

### E. FinTech, Legal & Regulated Enterprise (10 Companies)
| # | Company | Key Production RAG Pattern from Tech Blog | Status in Our Repo |
| :- | :--- | :--- | :--- |
| 41 | **Rippling** | **Eval-Driven Development (EDD)**: smoke evals on every commit, cost-uncertainty-lag | ✅ Implemented (Pytest + 5-mode ablation; add CI workflow) |
| 42 | **Bloomberg** | *EvidenT* provable source traceability + RAG Safety & Risk Taxonomies | ✅ Partial (Needs span-level exact quote highlighting) |
| 43 | **Capital One** | Platform-first governance + *FB-RAG* (Forward-Backward RAG for constraints) | 🔲 TODO (Cyclic self-correction / FB lookahead) |
| 44 | **JPMorgan Chase** | Regulated financial compliance, strict data lineage & auditability | 🔲 TODO (Immutable audit log for every query & doc) |
| 45 | **Ramp** | Auditable RAG grounding classification in verifiable rules & data lineage | 🔲 TODO (Structured JSON schema / rule-grounded mode) |
| 46 | **Thomson Reuters** | Legal two-stage bi-encoder + cross-encoder separating retrieval vs. generation evals | ✅ Implemented |
| 47 | **Plaid** | Regulated fintech data security & compliance-constrained retrieval | ✅ Partial |
| 48 | **Block** | *Labeled Queries* sidecar vector store (vetted human queries take priority over raw logs) | 🔲 TODO (Vetted Golden QA sidecar cache) |
| 49 | **Square** | Cosine similarity over domain embeddings for item/intent matching | ✅ Implemented |
| 50 | **Monzo** | Lightweight precomputed vector distance serving for fast help-center retrieval | ✅ Implemented |

---

## 2. Prioritized Project TODO Roadmap (Mapped to Our Codebase)

### P0 — High-Impact Retrieval & Agent Reliability (Interview Differentiators)
- [ ] **P0.1 Cyclic Self-Correction in LangGraph** (`backend/app/agent/graph.py`, `nodes.py`)
  - *Inspired by:* **Uber (Genie)**, **Capital One (FB-RAG)**
  - Add a cyclic conditional edge after `verify_faithfulness_self_rag`: if `faithfulness_score < 0.75` and `retry_count < 2`, broaden `top_k`, rewrite the query using critique feedback, and loop back to `retrieve_and_rerank`.
- [ ] **P0.2 Negation-Aware Reranking & MMR Diversity + Lost-in-the-Middle Reordering** (`backend/app/retrieval/postprocessor.py`)
  - *Inspired by:* **Grab**, **OpenAI**, **Cursor**
  - Add Maximal Marginal Relevance (MMR) deduplication to prevent near-duplicate chunks from consuming top-$k$ slots, U-shaped Lost-in-the-Middle reordering (`[Rank 1, Rank 3, ..., Rank 2]`), and explicit negation penalty detection.
- [ ] **P0.3 Table-Header & Code-Block-Aware Chunking + Optional LLM Contextualizer** (`backend/app/ingestion/chunking.py`)
  - *Inspired by:* **Anthropic**, **NVIDIA**, **Cursor**, **Atlassian**
  - Preserve Markdown table headers across split boundaries and keep fenced code blocks intact; add an optional Anthropic-style LLM contextual summary prefix during ingestion.
- [ ] **P0.4 Multi-Turn Conversational Memory & Coreference Resolution** (`backend/app/memory/conversation.py`)
  - *Inspired by:* **Booking.com**, **Lyft (AI Assist)**
  - Add `session_id` sliding-window memory that resolves pronouns (`"it"`, `"its remediation"`, `"that error"`) into standalone queries before retrieval.

### P1 — Production Operations, Observability & Compliance
- [ ] **P1.1 Circuit Breakers + `GET /metrics` (p50/p95/p99, Cache Hit Rate, Error Rate)** (`backend/app/observability/`)
  - *Inspired by:* **Cloudflare**, **Datadog**, **AWS**
  - Wrap Ollama & DuckDuckGo calls with a `CLOSED -> OPEN -> HALF_OPEN` circuit breaker and expose Prometheus/JSON latency & guardrail metrics at `GET /metrics`.
- [ ] **P1.2 GDPR Surgical Deletion (`DELETE /documents/{doc_id}`) & Version Precedence** (`backend/main.py`, `hybrid.py`)
  - *Inspired by:* **Figma**, **Notion**, **JPMorgan Chase**
  - Delete a single `doc_id` across Dense, BM25, Entity Graph, and Semantic Cache, and add `version` / `updated_at` recency weighting in RRF so newer docs outrank deprecated specs.
- [ ] **P1.3 Async Ingestion Queue (`rag_ingestion_jobs`)** (`backend/app/worker/consumer.py`)
  - *Inspired by:* **Cloudflare**, **Notion**, **AWS**
  - Offload heavy batch/PDF ingestion to RabbitMQ so `POST /ingest-async` returns a `job_id` immediately without blocking FastAPI workers.
- [ ] **P1.4 Negation & PII-Safe Audit Logging** (`backend/app/guardrails/audit.py`)
  - *Inspired by:* **Bloomberg**, **Ramp**, **JPMorgan Chase**
  - Emit structured, PII-redacted audit records linking every generated response to exact `doc_id`, `chunk_id`, and `user_roles`.

### P2 — Evaluation, Calibration & Ecosystem Integration
- [ ] **P2.1 LLM-as-a-Judge & Human-Calibrated Evals in `benchmark.py`**
  - *Inspired by:* **Booking.com**, **Dropbox**, **Twilio**, **NVIDIA**
  - Add optional local Ollama / OpenAI-compatible Judge scoring for Faithfulness & Answer Relevance alongside IR metrics (`HitRate@K`, `MRR@K`, `NDCG@K`).
- [ ] **P2.2 GitHub Actions CI Eval-Driven Development (EDD) Gate** (`.github/workflows/edd.yml`)
  - *Inspired by:* **Rippling**
  - Run `pytest` + the 5-mode ablation benchmark on every PR and fail if `MRR@3` or `HitRate@3` regresses.
- [ ] **P2.3 Expose RAG Engine as an MCP (Model Context Protocol) Server & Labeled-Query Sidecar**
  - *Inspired by:* **Expedia Group**, **Block**
  - Expose `retrieve_hybrid` and `ingest_document` as MCP tools and add a vetted "Labeled Queries" sidecar store for high-frequency head queries (**Instacart**, **Block**).
