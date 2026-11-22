"""FastAPI Gateway for the Staff-Level Production RAG System.

Orchestrates:
- Idempotent Parent-Child & Contextual Document Ingestion (SHA-256 content hashing)
- Two-Stage Hybrid Retrieval (Dense Milvus + Sparse BM25 via Reciprocal Rank Fusion + Cross-Encoder Reranking)
- Explicit LangGraph Corrective RAG (CRAG) + Self-RAG Faithfulness StateGraph
- Asynchronous RabbitMQ Job Queue with Local Thread Fallback
- Quantitative IR & RAG Ablation Evaluation Suite (HitRate@K, MRR@K, NDCG@K, ContextPrecision@K)
"""

from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.agent.graph import execute_rag_pipeline, stream_rag_pipeline_sse
from app.agent.sql_tool import sql_inventory_tool
from app.agent.structured import execute_structured_taxonomy_query
from app.agent.subagents import supervisor_orchestrator
from app.config import settings
from app.evaluation.benchmark import run_ablation_benchmark
from app.guardrails.audit import audit_logger
from app.ingestion.pipeline import ingestion_pipeline
from app.models import (
    CDCEventRequest,
    DocumentInput,
    EvaluationRequest,
    OrchestrateRequest,
    OrchestrateResponse,
    PrewarmCacheRequest,
    QueryInput,
    RetrieveDebugRequest,
    SQLQueryRequest,
    StructuredQueryRequest,
    StructuredQueryResponse,
    UserFeedbackRequest,
)
from app.observability.feedback import feedback_store
from app.observability.metrics import metrics_collector
from app.observability.tracing import trace_collector
from app.retrieval.embeddings import ModelRegistry
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.labeled_queries import labeled_query_store
from app.retrieval.prefix_cache import prefix_cache
from app.retrieval.semantic_cache import semantic_cache
from app.worker.consumer import (
    enqueue_ingestion_job,
    enqueue_rag_job,
    get_job_state,
    start_background_worker,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Starts the RabbitMQ background worker thread on FastAPI startup."""
    start_background_worker()
    yield


app = FastAPI(
    title="Staff RAG Engineering Platform",
    description=(
        "Production-grade Agentic & Corrective RAG (CRAG / Self-RAG) microservice featuring "
        "Two-Stage Hybrid Retrieval (Milvus Dense + BM25 RRF + Cross-Encoder Reranking) and built-in IR Evaluation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Returns system readiness, loaded singleton model status, and hybrid index statistics."""
    return {
        "status": "healthy",
        "models": ModelRegistry.status(),
        "corpus": retrieval_engine.stats(),
        "semantic_cache": semantic_cache.stats(),
        "config": {
            "milvus_host": settings.milvus_host,
            "rabbitmq_host": settings.rabbitmq_host,
            "llm_api_base": settings.llm_api_base,
            "relevance_threshold": settings.relevance_threshold,
            "rrf_k": settings.rrf_k,
        },
    }


@app.post("/ingest")
async def ingest_document(doc: DocumentInput):
    """Chunks, deduplicates (SHA-256), and indexes a document into Milvus and BM25."""
    if not doc.content or not doc.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    try:
        return ingestion_pipeline.ingest(doc)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.post("/ingest-async")
async def ingest_document_async(doc: DocumentInput):
    """Queues an asynchronous document ingestion job and returns a job_id."""
    if not doc.content or not doc.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    return enqueue_ingestion_job(doc)


@app.delete("/documents/{doc_id}")
async def gdpr_erase_document(doc_id: str):
    """GDPR Article 17 Right-to-be-Forgotten: surgically deletes doc_id across all stores + logs audit."""
    chunks_deleted = retrieval_engine.delete_by_doc_id(doc_id)
    cache_evicted = semantic_cache.invalidate_by_doc_id(doc_id)
    metrics_collector.record_gdpr_erasure()
    audit_event = audit_logger.log_event(
        event_type="gdpr_article17_erasure",
        doc_ids=[doc_id],
        summary=f"Erased doc_id={doc_id} ({chunks_deleted} chunks, {cache_evicted} cache entries)",
        metadata={"chunks_deleted": chunks_deleted, "cache_evicted": cache_evicted},
    )
    return {
        "status": "erased",
        "doc_id": doc_id,
        "chunks_deleted": chunks_deleted,
        "cache_entries_evicted": cache_evicted,
        "audit_event_hash": audit_event["event_hash"],
    }


@app.get("/metrics")
async def get_observability_metrics():
    """Returns rolling p50/p95/p99 latencies, cache hit rate, guardrail triggers, and circuit breakers."""
    return metrics_collector.snapshot()


@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Exposes RAG operational, latency, corpus, circuit breaker, and audit integrity metrics for Prometheus."""
    audit_state = audit_logger.get_logs(limit=1)
    prom_text = metrics_collector.format_prometheus(
        corpus_stats=retrieval_engine.stats(),
        chain_valid=bool(audit_state["chain_valid"]),
        trace_stats=trace_collector.stats(),
    )
    return PlainTextResponse(content=prom_text, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/traces")
async def get_distributed_traces(limit: int = 50):
    """Returns Langfuse-compatible distributed RAG traces and per-node span timings."""
    return trace_collector.list_traces(limit=limit)


@app.get("/traces/{trace_id}")
async def get_single_trace(trace_id: str):
    """Retrieves a single distributed RAG trace and its spans by trace_id."""
    rec = trace_collector.get_trace(trace_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Trace not found")
    return rec


@app.get("/documents")
async def list_indexed_documents():
    """Returns per-document inventory, RBAC role lineage, and chunk counts for audit inspection."""
    docs = retrieval_engine.list_documents()
    return {"count": len(docs), "documents": docs}


@app.get("/audit-logs")
async def get_audit_logs(limit: int = 100):
    """Returns tamper-evident SHA-256 hash-chained audit logs and chain integrity verification."""
    return audit_logger.get_logs(limit=limit)


@app.get("/compliance/audit-report")
async def get_compliance_audit_report(limit: int = 50):
    """Generates a complete System Compliance & Audit Dossier for regulatory/internal audits."""
    audit_data = audit_logger.get_logs(limit=limit)
    metrics_snap = metrics_collector.snapshot()
    docs = retrieval_engine.list_documents()
    traces_data = trace_collector.list_traces(limit=20)
    corpus_stats = retrieval_engine.stats()

    chain_badge = "VALID (Cryptographically Verified)" if audit_data["chain_valid"] else "TAMPERED (Verification Failed)"
    generated_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    lat = metrics_snap["latency_ms"]
    cbs = metrics_snap["circuit_breakers"]

    lines = [
        "# 🛡️ System Compliance & Observability Audit Dossier",
        f"- **Generated:** `{generated_utc}`",
        f"- **SHA-256 Audit Chain Integrity:** **{chain_badge}** (`{audit_data['total_events']}` total events, head=`{audit_data['head_hash'][:16]}...`)",
        f"- **GDPR Article 17 Erasures Executed:** `{metrics_snap['gdpr_erasures_total']}` | "
        f"**Security Guardrail Triggers (PII/Injection):** `{metrics_snap['guardrail_triggers_total']}`",
        f"- **Operational SLOs:** `total_queries={metrics_snap['total_queries']}`, "
        f"`p50={lat['p50']} ms`, `p95={lat['p95']} ms`, `p99={lat['p99']} ms`, "
        f"`cache_hit_rate={metrics_snap['cache_hit_rate']:.2%}`, `crag_fallback_rate={metrics_snap['crag_fallback_rate']:.2%}`",
        f"- **Downstream Circuit Breakers:** `ollama_llm={cbs['ollama_llm']['state']}`, `duckduckgo_web={cbs['duckduckgo_web']['state']}`",
        f"- **Langfuse Distributed Tracing:** `{traces_data['total_traces']}` traces recorded "
        f"(Langfuse external sync configured: `{traces_data['langfuse_configured']}`, synced: `{traces_data['langfuse_synced_count']}`)",
        "",
        f"## 📚 Active Corpus & RBAC Lineage (`{len(docs)}` Documents, `{corpus_stats.get('total_chunks', 0)}` Chunks)",
        "| Doc ID | Title | Tenant | Authorized Roles | Chunks | Parent Windows | Source |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :--- |",
    ]
    if not docs:
        lines.append("| - | _No active documents in corpus_ | - | - | 0 | 0 | - |")
    else:
        for d in docs:
            roles_str = ", ".join(d["access_roles"])  # type: ignore[arg-type]
            lines.append(
                f"| `{d['doc_id']}` | {d['title']} | `{d['tenant_id']}` | `{roles_str}` | "
                f"{d['chunk_count']} | {d['parent_windows_count']} | `{d['source']}` |"
            )

    gdpr_summary = {
        "article_17_erasures_total": metrics_snap["gdpr_erasures_total"],
        "pii_and_injection_guardrail_triggers": metrics_snap["guardrail_triggers_total"],
        "rbac_prefiltering_enabled": True,
        "audit_chain_valid": audit_data["chain_valid"],
        "active_documents_count": len(docs),
    }

    fb_summary = feedback_store.summary(limit=20)
    return {
        "generated_at_utc": generated_utc,
        "chain_valid": audit_data["chain_valid"],
        "head_hash": audit_data["head_hash"],
        "total_audit_events": audit_data["total_events"],
        "gdpr_compliance_summary": gdpr_summary,
        "feedback_summary": fb_summary,
        "metrics_snapshot": metrics_snap,
        "document_inventory": docs,
        "recent_traces": traces_data["traces"],
        "audit_events": audit_data["events"],
        "markdown_dossier": "\n".join(lines),
    }


@app.post("/ingest/pdf")
async def ingest_pdf_document(
    file: UploadFile = File(...),
    title: str = Form(default=""),
    tenant_id: str = Form(default="default"),
    access_roles: str = Form(default="public"),
    chunking_strategy: str = Form(default="parent_child"),
    enable_late_chunking: bool = Form(default=True),
    enable_contextual_enrichment: bool = Form(default=True),
):
    """#10 NVIDIA layout- & table-aware PDF parser + hybrid ingestion endpoint."""
    try:
        pdf_bytes = await file.read()
        roles = [r.strip().lower() for r in access_roles.split(",") if r.strip()] or ["public"]
        return ingestion_pipeline.ingest_pdf(
            pdf_bytes=pdf_bytes,
            filename=file.filename or "uploaded.pdf",
            title=title,
            tenant_id=tenant_id,
            access_roles=roles,
            chunking_strategy=chunking_strategy,
            enable_late_chunking=enable_late_chunking,
            enable_contextual_enrichment=enable_contextual_enrichment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {exc}") from exc


@app.post("/ingest/cdc")
async def ingest_cdc_webhook(event: CDCEventRequest):
    """Shopify/Notion-style CDC incremental upsert/delete endpoint keyed by external_id."""
    try:
        return ingestion_pipeline.process_cdc_event(event)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CDC event processing failed: {exc}") from exc


@app.post("/feedback")
async def record_user_feedback(req: UserFeedbackRequest):
    """Zendesk-style thumbs-up/down feedback endpoint tied to trace_id and retrieved chunk_ids."""
    entry = feedback_store.record(
        rating=req.rating,
        query=req.query,
        trace_id=req.trace_id,
        chunk_ids=req.chunk_ids,
        user_roles=req.user_roles,
        comment=req.comment,
    )
    audit_logger.log_event(
        event_type="user_feedback_recorded",
        actor_roles=req.user_roles,
        chunk_ids=req.chunk_ids,
        summary=f"User rated response '{req.rating}' for query '{req.query[:80]}'",
        metadata={"feedback_id": entry["feedback_id"], "trace_id": req.trace_id, "rating": req.rating},
    )
    return {"status": "recorded", "feedback": entry, "summary": feedback_store.summary(limit=5)}


@app.get("/feedback")
async def get_user_feedback_summary(limit: int = 25):
    """Returns rolling user satisfaction KPIs and recent thumbs-up/down feedback entries."""
    return feedback_store.summary(limit=limit)


@app.post("/ingest-sample")
async def ingest_sample_corpus(strategy: str = "parent_child"):
    """Seeds the built-in Enterprise Architecture & Governance sample corpus."""
    try:
        return ingestion_pipeline.ingest_sample_corpus(strategy=strategy)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sample ingestion failed: {exc}") from exc


@app.get("/corpus")
async def get_corpus_stats():
    """Returns current document, parent-window, and chunk statistics."""
    return retrieval_engine.stats()


@app.delete("/corpus")
async def clear_corpus():
    """Clears the in-memory hybrid index and semantic cache for clean benchmark resets."""
    retrieval_engine.clear()
    semantic_cache.clear()
    return {"status": "cleared", "corpus": retrieval_engine.stats()}


@app.get("/corpus/shards-and-clusters")
async def get_corpus_shards_and_clusters():
    """#14 Spotify Semantic ID clusters & #20 Airbnb tenant shard topology."""
    return retrieval_engine.get_shards_and_clusters()


@app.post("/retrieve")
async def debug_retrieve(req: RetrieveDebugRequest):
    """Directly executes Pre-Filtered RBAC + Tri-Modal Hybrid Retrieval without LLM generation."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    chunks = retrieval_engine.retrieve(
        queries=[req.query],
        mode=req.retrieval_mode,
        top_k=req.top_k,
        promote_parents=True,
        tenant_id=req.tenant_id,
        user_roles=req.user_roles,
        semantic_id_prefix=req.semantic_id_prefix,
    )
    return {
        "query": req.query,
        "retrieval_mode": req.retrieval_mode,
        "tenant_id": req.tenant_id,
        "user_roles": req.user_roles,
        "semantic_id_prefix": req.semantic_id_prefix,
        "count": len(chunks),
        "retrieval_funnel": retrieval_engine.get_last_funnel(),
        "results": [c.model_dump() for c in chunks],
    }


@app.post("/query-stream")
async def query_stream(query: QueryInput):
    """Streams real-time LangGraph node progression and final cited answer via Server-Sent Events (SSE)."""
    if not query.text or not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    return StreamingResponse(
        stream_rag_pipeline_sse(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/submit-job")
async def submit_job(query: QueryInput):
    """Enqueues an asynchronous RAG query job into RabbitMQ (with automatic local fallback)."""
    if not query.text or not query.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    try:
        return enqueue_rag_job(query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/get-job/{job_id}")
async def get_job(job_id: str):
    """Retrieves the status, cited answer, tool audit, and pipeline trace for a submitted job."""
    job = get_job_state(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/query-structured", response_model=StructuredQueryResponse)
async def query_structured_taxonomy(req: StructuredQueryRequest):
    """DoorDash Inverted RAG + Ramp Rule-Grounded structured JSON classification endpoint."""
    try:
        return execute_structured_taxonomy_query(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Structured query failed: {exc}") from exc


@app.post("/cache/prewarm")
async def prewarm_head_query_cache(req: PrewarmCacheRequest = PrewarmCacheRequest()):
    """#22 Instacart offline head-query cache pre-warming into role-scoped Semantic & Prefix Caches."""
    t0 = time.perf_counter()
    prewarmed = 0
    already_cached = 0
    for q_text in req.queries:
        if not q_text or not q_text.strip():
            continue
        out = execute_rag_pipeline(
            QueryInput(
                text=q_text.strip(),
                tenant_id=req.tenant_id,
                user_roles=req.user_roles,
                retrieval_mode=req.retrieval_mode,
                top_k=req.top_k,
                enable_web_search=False,
                enable_semantic_cache=True,
                max_retries=0,
            )
        )
        if out.get("trace", {}).get("cache_hit"):
            already_cached += 1
        else:
            prewarmed += 1

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    audit_logger.log_event(
        event_type="cache_prewarmed",
        actor_tenant=req.tenant_id,
        actor_roles=req.user_roles,
        summary=f"Pre-warmed {prewarmed} head queries ({already_cached} already cached) in {elapsed_ms}ms",
        metadata={"prewarmed_count": prewarmed, "already_cached_count": already_cached},
    )
    return {
        "status": "success",
        "prewarmed_count": prewarmed,
        "already_cached_count": already_cached,
        "total_latency_ms": elapsed_ms,
        "cache_stats": {
            "semantic_cache": semantic_cache.stats(),
            "prefix_cache": prefix_cache.stats(),
        },
    }


@app.get("/cache/stats")
async def get_cache_statistics():
    """Returns combined Semantic Query Cache & LinkedIn Shared-Prefix Prompt Cache telemetry."""
    return {
        "semantic_cache": semantic_cache.stats(),
        "prefix_cache": prefix_cache.stats(),
    }


@app.get("/inventory/sql-status")
async def get_live_sql_inventory_status():
    """#25 Agoda OpsBot live SQL service inventory & active incidents snapshot."""
    return sql_inventory_tool.get_live_snapshot()


@app.post("/inventory/sql-query")
async def run_readonly_sql_inventory_query(req: SQLQueryRequest):
    """#25 Agoda OpsBot strictly read-only SELECT SQL query endpoint."""
    try:
        return sql_inventory_tool.execute_readonly_sql(req.sql)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SQL query failed: {exc}") from exc


@app.get("/agent/subagents")
async def list_specialist_subagents():
    """P3.3 Lists registered least-privilege specialist subagents and their scoped toolsets."""
    subs = supervisor_orchestrator.list_subagents()
    return {"count": len(subs), "subagents": subs}


@app.post("/agent/orchestrate", response_model=OrchestrateResponse)
async def orchestrate_specialist_subagents(req: OrchestrateRequest):
    """P3.3 Parallel Supervisor orchestration across retrieval, SQL OpsBot, compliance auditor, and web specialists."""
    try:
        return supervisor_orchestrator.orchestrate(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Multi-agent orchestration failed: {exc}") from exc


@app.post("/evaluate")
async def evaluate_rag_pipeline(req: EvaluationRequest = EvaluationRequest()):
    """Executes the Golden QA Ablation Study across Dense, Sparse, Hybrid RRF, and Hybrid + Reranker."""
    try:
        return run_ablation_benchmark(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
