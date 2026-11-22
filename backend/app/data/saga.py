"""P4.3 Saga Pattern Orchestrator with Automatic Reverse Compensating Transactions."""

from __future__ import annotations

from dataclasses import dataclass
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from app.data.cqrs import cqrs_read_model
from app.guardrails.audit import audit_logger
from app.guardrails.security import guardrails
from app.ingestion.chunking import ChunkingEngine
from app.models import DocumentInput
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.semantic_cache import semantic_cache


@dataclass
class SagaStep:
    """One local transaction step and its idempotent compensating rollback action."""

    name: str
    action: Callable[[Dict[str, Any]], None]
    compensate: Optional[Callable[[Dict[str, Any]], None]] = None


class SagaOrchestrator:
    """Executes a sequence of local transactions; runs compensating rollbacks in reverse order on failure."""

    def run(
        self,
        steps: List[SagaStep],
        ctx: Dict[str, Any],
        inject_failure_step: Optional[str] = None,
    ) -> Dict[str, Any]:
        saga_id = f"saga_{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()
        completed: List[SagaStep] = []
        step_log: List[Dict[str, Any]] = []
        compensated_steps: List[str] = []

        try:
            for step in steps:
                if inject_failure_step and step.name == inject_failure_step:
                    raise RuntimeError(f"Injected saga failure at step '{step.name}'")
                step.action(ctx)
                completed.append(step)
                step_log.append({"step": step.name, "status": "completed"})

            return {
                "saga_id": saga_id,
                "saga_status": "committed",
                "failed_step": None,
                "error": None,
                "step_log": step_log,
                "compensated_steps": [],
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "context": ctx,
            }
        except Exception as exc:
            failed_step = steps[len(completed)].name if len(completed) < len(steps) else "unknown"
            step_log.append({"step": failed_step, "status": "failed", "error": str(exc)})
            for prev in reversed(completed):
                if prev.compensate is not None:
                    try:
                        prev.compensate(ctx)
                        compensated_steps.append(prev.name)
                        step_log.append({"step": prev.name, "status": "compensated"})
                    except Exception as comp_exc:
                        step_log.append({"step": prev.name, "status": "compensation_error", "error": str(comp_exc)})

            return {
                "saga_id": saga_id,
                "saga_status": "compensated",
                "failed_step": failed_step,
                "error": str(exc),
                "step_log": step_log,
                "compensated_steps": compensated_steps,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "context": ctx,
            }


_chunker = ChunkingEngine()
_orchestrator = SagaOrchestrator()


def execute_ingestion_saga(
    doc_input: DocumentInput,
    inject_failure_step: Optional[str] = None,
) -> Dict[str, Any]:
    """Coordinates multi-store ingestion (Chunk -> Vector/BM25/Graph -> CQRS -> Audit) with full rollback on error."""
    ctx: Dict[str, Any] = {"doc_input": doc_input}

    def step_chunk_and_sanitize(c: Dict[str, Any]) -> None:
        raw = doc_input.content
        flags: List[str] = []
        if doc_input.apply_guardrails:
            raw, flags = guardrails.sanitize_text(raw)
        meta = dict(doc_input.metadata or {})
        meta.setdefault("version", doc_input.version)
        meta["updated_at"] = float(doc_input.updated_at if doc_input.updated_at is not None else time.time())
        chunks = _chunker.chunk_document(
            content=raw,
            title=doc_input.title,
            source=doc_input.source,
            strategy=doc_input.chunking_strategy,
            tenant_id=doc_input.tenant_id,
            access_roles=doc_input.access_roles,
            extra_metadata=meta,
            enable_contextual_enrichment=doc_input.enable_contextual_enrichment,
            use_llm_contextualizer=doc_input.use_llm_contextualizer,
        )
        c["sanitized_content"] = raw
        c["guardrail_flags"] = flags
        c["chunks"] = chunks
        c["doc_id"] = chunks[0].doc_id if chunks else None

    def step_index_vector_and_lexical(c: Dict[str, Any]) -> None:
        added, skipped = retrieval_engine.add_chunks(
            c["chunks"],
            document_context_text=c["sanitized_content"] if doc_input.enable_late_chunking else None,
        )
        c["added"] = added
        c["skipped"] = skipped
        if c.get("doc_id"):
            semantic_cache.invalidate_by_doc_id(c["doc_id"])

    def compensate_vector_and_lexical(c: Dict[str, Any]) -> None:
        if c.get("doc_id"):
            retrieval_engine.delete_by_doc_id(c["doc_id"])
            semantic_cache.invalidate_by_doc_id(c["doc_id"])

    def step_update_cqrs_projection(c: Dict[str, Any]) -> None:
        if c.get("doc_id"):
            cqrs_read_model.project_document_upserted(
                doc_id=c["doc_id"],
                title=doc_input.title,
                source=doc_input.source,
                tenant_id=doc_input.tenant_id,
                access_roles=doc_input.access_roles,
                chunk_count=len(c.get("chunks") or []),
            )

    def compensate_cqrs_projection(c: Dict[str, Any]) -> None:
        if c.get("doc_id"):
            cqrs_read_model.remove_document(c["doc_id"])

    def step_record_audit_lineage(c: Dict[str, Any]) -> None:
        ev = audit_logger.log_event(
            event_type="saga_document_ingested",
            actor_tenant=doc_input.tenant_id,
            actor_roles=doc_input.access_roles,
            doc_ids=[c["doc_id"]] if c.get("doc_id") else [],
            summary=f"Saga committed '{doc_input.title}' ({c.get('added', 0)} chunks)",
        )
        c["audit_event_hash"] = ev["event_hash"]

    steps = [
        SagaStep("chunk_and_sanitize", step_chunk_and_sanitize),
        SagaStep("index_vector_and_lexical", step_index_vector_and_lexical, compensate_vector_and_lexical),
        SagaStep("update_cqrs_projection", step_update_cqrs_projection, compensate_cqrs_projection),
        SagaStep("record_audit_lineage", step_record_audit_lineage),
    ]

    result = _orchestrator.run(steps, ctx, inject_failure_step=inject_failure_step)
    if result["saga_status"] == "compensated":
        audit_logger.log_event(
            event_type="saga_compensated",
            actor_tenant=doc_input.tenant_id,
            actor_roles=doc_input.access_roles,
            doc_ids=[ctx["doc_id"]] if ctx.get("doc_id") else [],
            summary=(
                f"Saga {result['saga_id']} failed at '{result['failed_step']}' "
                f"and compensated {result['compensated_steps']}"
            ),
            metadata={"saga_id": result["saga_id"], "compensated_steps": result["compensated_steps"]},
        )

    return {
        "saga_id": result["saga_id"],
        "saga_status": result["saga_status"],
        "doc_id": ctx.get("doc_id"),
        "failed_step": result["failed_step"],
        "error": result["error"],
        "step_log": result["step_log"],
        "compensated_steps": result["compensated_steps"],
        "duration_ms": result["duration_ms"],
        "chunks_indexed_new": ctx.get("added", 0) if result["saga_status"] == "committed" else 0,
        "audit_event_hash": ctx.get("audit_event_hash"),
    }
