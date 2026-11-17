# 🏗️ Staff RAG Engineering Platform (Microservices, Hybrid RRF, Cross-Encoder & LangGraph CRAG/Self-RAG)

This repository implements an interview-ready, production-grade **Retrieval-Augmented Generation (RAG)** platform orchestrated with **Docker Compose**, **FastAPI**, **Milvus**, **RabbitMQ**, **LangGraph**, **Ollama**, and **Gradio**.

Designed as a **Staff RAG Engineer Showcase**, it goes beyond naive dense-only prototypes by implementing **Small-to-Big Parent-Child Chunking**, **SHA-256 Content-Addressable Deduplication**, **Two-Stage Hybrid Retrieval (Dense Milvus + Sparse BM25 via Reciprocal Rank Fusion + Cross-Encoder Reranking)**, an explicit **Corrective RAG (CRAG) & Self-RAG LangGraph StateGraph**, and a built-in **Quantitative IR Ablation Benchmark** (`HitRate@K`, `MRR@K`, `NDCG@K`, `ContextPrecision@K`).

📖 **Preparing for an interview?** Read the companion [**Staff RAG Interview Guide (`docs/INTERVIEW_GUIDE.md`)**](docs/INTERVIEW_GUIDE.md).

---

## 🏛️ System Architecture

| Service | Technology | Internal Port | External Port | Role in RAG Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Gradio Studio** | Gradio | `7860` | **7860** | 3-Tab UI: Ingestion Studio, Async Query & Trace Inspector, and IR Ablation Runner. |
| **FastAPI Gateway & Worker** | FastAPI + LangGraph | `8000` | **8000** | Modular RAG engine (`backend/app/`), hybrid retriever, Cross-Encoder reranker, and RabbitMQ worker. |
| **LLM Server** | Ollama (`llama3.1`) | `11434` | **11434** | Local OpenAI-compatible inference endpoint for cited synthesis. |
| **Vector DB** | Milvus Standalone | `19530` | **19530** | Stores 384-dim dense bi-encoder embeddings (`all-MiniLM-L6-v2`) with Strong consistency. |
| **Message Broker** | RabbitMQ | `5672` | **15672** (UI) | Decouples async RAG query execution (`rag_jobs` queue) with automatic local thread fallback. |
| **Vector Admin UI** | Zilliz Attu | `3000` | **9090** | Visual inspector for Milvus collections, schemas, and entities. |
| **Object & Meta Storage** | MinIO & Etcd | `9000` / `2379` | Internal | Storage and distributed coordination dependencies for Milvus. |

---

## 📂 Modular Project Structure

```text
RAG/
├── backend/
│   ├── app/
│   │   ├── config.py              # Environment-backed settings (fixes legacy host.docker.internal bug)
│   │   ├── models.py              # Pydantic schemas for chunks, traces, and evaluation reports
│   │   ├── ingestion/
│   │   │   ├── chunking.py        # Parent-Child (Small-to-Big), Contextual Header & SHA-256 dedup
│   │   │   └── pipeline.py        # Idempotent Milvus + BM25 indexing & Enterprise Sample Corpus
│   │   ├── retrieval/
│   │   │   ├── embeddings.py      # Thread-safe singleton registry (Bi-Encoder, Cross-Encoder, LLM)
│   │   │   ├── hybrid.py          # Dense Milvus + Sparse BM25 + Reciprocal Rank Fusion (RRF, k=60)
│   │   │   └── reranker.py        # Cross-Encoder (ms-marco-MiniLM-L-6-v2) + Sigmoid Calibration
│   │   ├── agent/
│   │   │   ├── state.py           # Typed LangGraph state schema
│   │   │   ├── nodes.py           # Query expander, Hybrid retriever, CRAG gate, Web search, Self-RAG grader
│   │   │   └── graph.py           # Compiled LangGraph StateGraph workflow
│   │   ├── evaluation/
│   │   │   ├── metrics.py         # HitRate@K, MRR@K, NDCG@K, ContextPrecision@K
│   │   │   └── benchmark.py       # Golden Technical QA Benchmark & 4-mode ablation study
│   │   └── worker/
│   │       └── consumer.py        # RabbitMQ producer/consumer with in-process thread fallback
│   ├── tests/
│   │   └── test_rag_pipeline.py   # Automated unit & integration test suite
│   ├── main.py                    # FastAPI entrypoint (/health, /ingest, /retrieve, /submit-job, /evaluate)
│   ├── debug.py                   # CLI diagnostic & ablation inspector
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                     # 3-Tab Gradio Observability & Evaluation Studio
│   ├── requirements.txt           # Lightweight UI dependencies (gradio, requests, pypdf)
│   └── Dockerfile
├── docs/
│   └── INTERVIEW_GUIDE.md         # Staff RAG interview prep guide & mathematical walkthrough
├── graphify-out/                  # Persistent codebase knowledge graph (graph.html, GRAPH_REPORT.md)
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
*(Note: Even before `llama3.1` finishes downloading, the pipeline includes a deterministic grounded synthesis fallback so you can test ingestion, hybrid retrieval, Cross-Encoder reranking, and the Ablation Benchmark immediately!)*

### 3. Open the Interfaces
- **Gradio RAG Studio:** [http://localhost:7860](http://localhost:7860)
- **FastAPI Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **RabbitMQ Management UI:** [http://localhost:15672](http://localhost:15672) (`guest` / `guest`)
- **Attu Milvus UI:** [http://localhost:9090](http://localhost:9090)

---

## 🧪 Running Tests & CLI Diagnostics Locally

Run the automated test suite or the CLI ablation inspector directly:

```bash
cd backend
pytest -v
python debug.py
```
