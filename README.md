# 🏗️ Staff+ / Principal RAG Engineering Platform

An interview-ready, production-grade **Retrieval-Augmented Generation (RAG)** microservices platform orchestrated with **Docker Compose**, **FastAPI**, **Milvus**, **RabbitMQ**, **LangGraph**, **Ollama**, and **Gradio**.

Designed as a **Staff+ / Principal RAG Engineer Showcase**, it implements real-world patterns synthesized from 50+ industry engineering blogs ([`docs/INDUSTRY_RAG_TODO.md`](docs/INDUSTRY_RAG_TODO.md)), including **Pre-Filtered RBAC**, **Tri-Modal GraphRAG**, **Cyclic LangGraph Self-Correction**, **GDPR Article 17 Surgical Erasure**, and a **Tamper-Evident SHA-256 Audit Trail**.

📖 **Interview & Architecture Guides:**
- [**Staff RAG Interview Guide (`docs/INTERVIEW_GUIDE.md`)**](docs/INTERVIEW_GUIDE.md) — Math formulations (RRF, Cross-Encoder Calibration, NDCG), 6 legacy anti-patterns fixed, and system design Q&A.
- [**50-Company RAG Tech Blog Study & Roadmap (`docs/INDUSTRY_RAG_TODO.md`)**](docs/INDUSTRY_RAG_TODO.md) — Production lessons from Google, Netflix, Anthropic, Uber, DoorDash, Bloomberg, and more.

---

## ✨ Key Production Capabilities

- **Pre-Filtered Multi-Tenant RBAC (`tenant_id`, `access_roles`):** Enforces role and tenant isolation *before* Dense, Sparse, Graph, and Cross-Encoder scoring—eliminating data leakage and post-filter recall starvation.
- **Tri-Modal Hybrid Retrieval & Post-Processing:**
  - **Stage 1 Recall:** Milvus Dense Bi-Encoder (`all-MiniLM-L6-v2`) + Sparse `BM25Okapi` + 1-hop Entity Knowledge Graph fused via **Reciprocal Rank Fusion (`RRF_K = 60`)**.
  - **Stage 2 Precision:** **Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`)** with sigmoid calibration, negation-polarity adjustment, **Maximal Marginal Relevance (MMR)** diversity selection, and **Lost-in-the-Middle** U-shaped reordering.
- **Table-Header-Preserving & Parent-Child Chunking:** Indexes fine-grained child chunks (`380` chars) with contextual headers, re-injects Markdown table headers across splits, promotes deduplicated parent windows (`1200` chars), and deduplicates idempotently via SHA-256 chunk IDs.
- **Cyclic LangGraph (`CRAG` + `Self-RAG`) & Adaptive Query Router:**
  - Routes queries dynamically (`lexical_lookup`, `conceptual_hyde`, `multi_hop`), resolves multi-turn pronouns via `session_id` memory, gates irrelevant context via CRAG, and loops back to `retrieve_and_rerank` if Self-RAG faithfulness falls below `0.75`.
  - Streams node transitions live via **Server-Sent Events (`POST /query-stream`)** or processes asynchronously via **RabbitMQ** (`POST /submit-job`, `POST /ingest-async`).
- **Role-Scoped Semantic Cache & Labeled-Query Sidecar:** Sub-millisecond cache hits scoped by `(tenant_id, user_roles, retrieval_mode)` with automatic `doc_id` invalidation on document updates.
- **GDPR, Security Guardrails & Tamper-Evident Auditability:**
  - Redacts PII (emails, SSNs, API keys) and blocks indirect prompt injections on ingestion and query paths.
  - Supports **GDPR Article 17 Right-to-be-Forgotten** (`DELETE /documents/{doc_id}`) across Dense, BM25, Entity Graph, and Semantic Cache.
  - Records every action in a **SHA-256 hash-chained immutable audit log** (`GET /audit-logs`) and exposes **Circuit Breakers** + `p50`/`p95`/`p99` telemetry at `GET /metrics`.
- **MCP Server & CI Eval-Driven Development (EDD):** Exposes RBAC-scoped RAG tools in [`backend/mcp_server.py`](backend/mcp_server.py) and enforces retrieval regression gates in [`.github/workflows/edd.yml`](.github/workflows/edd.yml).

---

## 🏛️ Microservices Architecture

| Service | Technology | Internal Port | External Port | Role in RAG Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Gradio Studio** | Gradio | `7860` | **7860** | 3-Tab UI: Ingestion/RBAC Studio, Live SSE & RabbitMQ Query Studio, 5-Mode Ablation Runner. |
| **FastAPI Gateway & Worker** | FastAPI + LangGraph | `8000` | **8000** | Modular RAG engine (`backend/app/`), hybrid/graph retriever, guardrails, and worker pool. |
| **LLM Server** | Ollama (`llama3.1`) | `11434` | **11434** | Local OpenAI-compatible inference endpoint for cited synthesis. |
| **Vector DB** | Milvus Standalone | `19530` | **19530** | Stores 384-dim dense bi-encoder embeddings with Strong consistency. |
| **Message Broker** | RabbitMQ | `5672` | **15672** (UI) | Decouples async query & ingestion jobs with automatic local thread fallback. |
| **Vector Admin UI** | Zilliz Attu | `3000` | **9090** | Visual inspector for Milvus collections, schemas, and entities. |
| **Object & Meta Storage** | MinIO & Etcd | `9000` / `2379` | Internal | Storage and distributed coordination dependencies for Milvus. |

---

## 📂 Modular Project Structure

```text
RAG/
├── .github/workflows/edd.yml      # CI Eval-Driven Development (pytest + 5-mode ablation gate)
├── backend/
│   ├── app/
│   │   ├── config.py              # Environment-backed settings (Ollama, Milvus, RabbitMQ, thresholds)
│   │   ├── models.py              # Pydantic contracts (RBAC, GDPR metadata, PipelineTrace, Eval)
│   │   ├── ingestion/
│   │   │   ├── chunking.py        # Parent-Child, Table-Header preservation & SHA-256 deduplication
│   │   │   └── pipeline.py        # Idempotent ingestion + cache invalidation + GDPR audit logging
│   │   ├── retrieval/
│   │   │   ├── embeddings.py      # Thread-safe singleton registry (Bi-Encoder, Cross-Encoder, LLM)
│   │   │   ├── entity_graph.py    # 1-hop Entity Knowledge Graph for Tri-Modal GraphRAG
│   │   │   ├── hybrid.py          # Pre-Filtered RBAC + Dense + BM25 + Graph RRF + GDPR erasure
│   │   │   ├── reranker.py        # Cross-Encoder (ms-marco-MiniLM-L-6-v2) + Sigmoid Calibration
│   │   │   ├── postprocessor.py   # MMR diversity, negation adjustment & Lost-in-the-Middle reorder
│   │   │   ├── semantic_cache.py  # Role-scoped Semantic Query Cache with doc_id invalidation
│   │   │   └── labeled_queries.py # Vetted Golden QA sidecar cache
│   │   ├── agent/
│   │   │   ├── state.py           # Typed LangGraph state (routing, retries, critiques, telemetry)
│   │   │   ├── nodes.py           # Adaptive Router, CRAG gate, Circuit-Broken Web/LLM, Self-RAG
│   │   │   └── graph.py           # Cyclic LangGraph StateGraph + SSE streaming generator
│   │   ├── guardrails/
│   │   │   ├── security.py        # PII redaction & indirect prompt injection defense
│   │   │   └── audit.py           # Tamper-evident SHA-256 hash-chained audit trail
│   │   ├── memory/
│   │   │   └── conversation.py    # Multi-turn session memory & pronoun coreference resolution
│   │   ├── observability/
│   │   │   └── metrics.py         # Circuit Breakers + rolling p50/p95/p99 latency & counter metrics
│   │   ├── evaluation/
│   │   │   ├── metrics.py         # HitRate@K, MRR@K, NDCG@K, ContextPrecision@K
│   │   │   └── benchmark.py       # 5-Mode Golden Technical QA Ablation Benchmark
│   │   └── worker/
│   │       └── consumer.py        # Async RabbitMQ query & ingestion workers with thread fallback
│   ├── mcp_server.py              # RBAC-scoped MCP server (retrieve, ingest, GDPR erase)
│   ├── main.py                    # FastAPI gateway (/ingest, /query-stream, /documents/{id}, /metrics, /audit-logs)
│   ├── debug.py                   # CLI diagnostic & ablation inspector
│   ├── tests/                     # 23 automated TDD unit, integration, GDPR & OWASP security tests
│   └── Dockerfile
├── frontend/
│   ├── app.py                     # 3-Tab Gradio Studio (Ingestion/RBAC, SSE Query, 5-Mode Eval)
│   ├── requirements.txt           # Lightweight UI dependencies (gradio, huggingface_hub<1.0, requests, pypdf)
│   └── Dockerfile
├── docs/
│   ├── INTERVIEW_GUIDE.md
│   └── INDUSTRY_RAG_TODO.md
└── docker-compose.yml
```

---

## 🚀 Quick Start (Docker Compose)

### 1. Build and Launch All Containers
```bash
docker-compose up -d --build
```

### 2. Pull the Local LLM inside Ollama
```bash
docker exec -it ollama ollama pull llama3.1
```

### 3. Access the Interfaces
- **Gradio Studio UI:** [http://localhost:7860](http://localhost:7860)
- **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Observability Metrics (`p50`/`p95`/`p99`):** [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Tamper-Evident Audit Logs:** [http://localhost:8000/audit-logs](http://localhost:8000/audit-logs)
- **RabbitMQ Management UI:** [http://localhost:15672](http://localhost:15672) (`guest` / `guest`)
- **Attu Milvus UI:** [http://localhost:9090](http://localhost:9090)

---

## 🧪 Running Tests & Diagnostics Locally

```bash
$env:PYTHONPATH = "$PWD\backend"
uv run --with fastapi --with pydantic --with pytest --with httpx --with rank-bm25 pytest backend/tests/ -v
uv run --with fastapi --with pydantic --with rank-bm25 python backend/debug.py
```
