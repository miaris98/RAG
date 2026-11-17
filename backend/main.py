"""FastAPI Gateway for the Staff-Level Production RAG System.

Orchestrates:
- Idempotent Parent-Child & Contextual Document Ingestion (SHA-256 content hashing)
- Two-Stage Hybrid Retrieval (Dense Milvus + Sparse BM25 via Reciprocal Rank Fusion + Cross-Encoder Reranking)
- Explicit LangGraph Corrective RAG (CRAG) + Self-RAG Faithfulness StateGraph
- Asynchronous RabbitMQ Job Queue with Local Thread Fallback
- Quantitative IR & RAG Ablation Evaluation Suite (HitRate@K, MRR@K, NDCG@K, ContextPrecision@K)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.config import settings
from app.evaluation.benchmark import run_ablation_benchmark
from app.ingestion.pipeline import ingestion_pipeline
from app.models import (
    DocumentInput,
    EvaluationRequest,
    QueryInput,
    RetrieveDebugRequest,
)
from app.retrieval.embeddings import ModelRegistry
from app.retrieval.hybrid import retrieval_engine
from app.worker.consumer import enqueue_rag_job, get_job_state, start_background_worker


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
    """Clears the in-memory hybrid index for clean benchmark resets."""
    retrieval_engine.clear()
    return {"status": "cleared", "corpus": retrieval_engine.stats()}


@app.post("/retrieve")
async def debug_retrieve(req: RetrieveDebugRequest):
    """Directly executes Two-Stage Hybrid Retrieval + Reranking for inspection without LLM generation."""
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    chunks = retrieval_engine.retrieve(
        queries=[req.query],
        mode=req.retrieval_mode,
        top_k=req.top_k,
        promote_parents=True,
    )
    return {
        "query": req.query,
        "retrieval_mode": req.retrieval_mode,
        "count": len(chunks),
        "results": [c.model_dump() for c in chunks],
    }


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


@app.post("/evaluate")
async def evaluate_rag_pipeline(req: EvaluationRequest = EvaluationRequest()):
    """Executes the Golden QA Ablation Study across Dense, Sparse, Hybrid RRF, and Hybrid + Reranker."""
    try:
        return run_ablation_benchmark(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
