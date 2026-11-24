"""P3.3: Specialist Subagents & Parallel Supervisor Multi-Agent Orchestrator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
import time
from typing import Any, Dict, List, Optional

from app.agent.nodes import _build_structured_citations, _format_citations_bibliography
from app.agent.sql_tool import LIVE_SQL_TRIGGER_RE, sql_inventory_tool
from app.agent.structured import execute_structured_taxonomy_query
from app.evaluation.judge import rag_judge
from app.guardrails.audit import audit_logger
from app.guardrails.security import guardrails
from app.models import (
    CitationItem,
    OrchestrateRequest,
    OrchestrateResponse,
    StructuredQueryRequest,
    SubagentResult,
)
from app.observability.metrics import web_circuit_breaker
from app.observability.tracing import trace_collector
from app.retrieval.hybrid import retrieval_engine


DEFAULT_TAXONOMY_LABELS = [
    "payment_ledger_quorum_incident",
    "hsm_key_rotation_security",
    "data_retention_encryption_policy",
    "sovereign_compliance_escalation",
    "rag_architecture_slo",
]

SUBAGENT_REGISTRY: List[Dict[str, Any]] = [
    {
        "agent_id": "retrieval_specialist",
        "role": "Hybrid Knowledge Retrieval Specialist",
        "allowed_tools": ["retrieve_hybrid", "entity_graph_traverse", "cross_encoder_rerank"],
        "isolation_boundary": "Strict caller (tenant_id, user_roles) RBAC pre-filtering; read-only.",
    },
    {
        "agent_id": "sql_ops_specialist",
        "role": "Live Operational SQL Inventory Specialist (OpsBot)",
        "allowed_tools": ["execute_readonly_sql", "get_live_snapshot"],
        "isolation_boundary": "Read-only SQLite operational tables only; no vector/compliance corpus access.",
    },
    {
        "agent_id": "compliance_auditor",
        "role": "Compliance, Guardrails & Faithfulness Auditor",
        "allowed_tools": ["sanitize_text", "classify_structured_taxonomy", "rag_judge_evaluate"],
        "isolation_boundary": "Audits synthesized claims, enforces taxonomy bounds, flags hallucinations.",
    },
    {
        "agent_id": "web_researcher",
        "role": "External Web Research Specialist",
        "allowed_tools": ["duckduckgo_search"],
        "isolation_boundary": "Circuit-breaker guarded external search; no internal corpus access.",
    },
]


def _run_retrieval_specialist(req: OrchestrateRequest, clean_query: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    chunks = retrieval_engine.retrieve(
        queries=[clean_query],
        mode=req.retrieval_mode,
        top_k=req.top_k,
        promote_parents=True,
        tenant_id=req.tenant_id,
        user_roles=req.user_roles,
    )
    chunk_dicts = [c.model_dump() for c in chunks]
    relevant = [c for c in chunk_dicts if c.get("is_relevant")]
    citations_raw = _build_structured_citations(clean_query, relevant)

    summary_lines: List[str] = []
    for idx, chk in enumerate(relevant, start=1):
        clean_body = re.sub(r"^\[(?:Document|Context):.*?\]\n?", "", chk.get("parent_content", ""), flags=re.MULTILINE).strip()
        summary_lines.append(f"- [Doc {idx}: {chk['title']} §{chk.get('section', 'General')}]: {clean_body}")

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "result": SubagentResult(
            agent_id="retrieval_specialist",
            role="Hybrid Knowledge Retrieval Specialist",
            status="completed",
            latency_ms=latency_ms,
            tools_used=["retrieve_hybrid"],
            summary=(
                f"Retrieved {len(relevant)}/{len(chunk_dicts)} relevant chunks under roles={req.user_roles}."
            ),
            output_preview="\n".join(summary_lines)[:400],
        ),
        "relevant_chunks": relevant,
        "citations_raw": citations_raw,
        "internal_markdown": "\n".join(summary_lines),
    }


def _run_sql_ops_specialist(clean_query: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    sql_payload = sql_inventory_tool.maybe_answer_from_live_sql(clean_query)
    if sql_payload is None:
        snap = sql_inventory_tool.get_live_snapshot()
        sql_payload = {
            "snapshot": snap,
            "markdown": f"Live SQL snapshot: {len(snap['services'])} services, {len(snap['active_incidents'])} active incidents.",
        }
    snap = sql_payload["snapshot"]
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "result": SubagentResult(
            agent_id="sql_ops_specialist",
            role="Live Operational SQL Inventory Specialist (OpsBot)",
            status="completed",
            latency_ms=latency_ms,
            tools_used=["execute_readonly_sql"],
            summary=f"Queried live SQL DB: {len(snap['services'])} services, {len(snap['active_incidents'])} incidents.",
            output_preview=sql_payload["markdown"][:400],
        ),
        "sql_markdown": sql_payload["markdown"],
    }


def _run_web_researcher(clean_query: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    web_text = "External web search skipped (circuit open or offline)."
    status = "fallback"
    if web_circuit_breaker.allow_request():
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                hits = list(ddgs.text(clean_query, max_results=2))
            if hits:
                web_text = "\n".join(f"- {h.get('title', '')}: {h.get('body', '')[:160]}" for h in hits)
                status = "completed"
                web_circuit_breaker.record_success()
        except Exception:
            web_circuit_breaker.record_failure()
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "result": SubagentResult(
            agent_id="web_researcher",
            role="External Web Research Specialist",
            status=status,
            latency_ms=latency_ms,
            tools_used=["duckduckgo_search"],
            summary=f"External web research status={status}.",
            output_preview=web_text[:300],
        ),
        "web_markdown": web_text,
    }


class SupervisorOrchestrator:
    """Decomposes complex enterprise queries across parallel least-privilege Specialist Subagents."""

    def list_subagents(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in SUBAGENT_REGISTRY]

    def orchestrate(self, req: OrchestrateRequest) -> OrchestrateResponse:
        t_start = time.perf_counter()
        clean_query, guardrail_flags = guardrails.sanitize_text(req.query.strip())

        need_sql = bool(req.force_all_subagents or LIVE_SQL_TRIGGER_RE.search(clean_query))
        need_web = bool(req.enable_web_search)

        # Stage 1: Parallel fan-out across evidence-gathering specialist subagents
        subagent_results: List[SubagentResult] = []
        retrieval_data: Dict[str, Any] = {}
        sql_data: Optional[Dict[str, Any]] = None
        web_data: Optional[Dict[str, Any]] = None

        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_ret = pool.submit(_run_retrieval_specialist, req, clean_query)
            fut_sql = pool.submit(_run_sql_ops_specialist, clean_query) if need_sql else None
            fut_web = pool.submit(_run_web_researcher, clean_query) if need_web else None

            retrieval_data = fut_ret.result()
            subagent_results.append(retrieval_data["result"])
            if fut_sql is not None:
                sql_data = fut_sql.result()
                subagent_results.append(sql_data["result"])
            if fut_web is not None:
                web_data = fut_web.result()
                subagent_results.append(web_data["result"])

        # Stage 2: Supervisor grounded synthesis combining specialist evidence
        sections: List[str] = ["### Multi-Agent Supervisor Synthesis"]
        if retrieval_data.get("internal_markdown"):
            sections.append(f"#### Internal Corpus Evidence (`retrieval_specialist`)\n{retrieval_data['internal_markdown']}")
        else:
            sections.append(
                "#### Internal Corpus Evidence (`retrieval_specialist`)\n"
                "No authorized internal documents passed the relevance threshold for your RBAC roles."
            )

        if sql_data is not None:
            sections.append(sql_data["sql_markdown"])
        if web_data is not None:
            sections.append(f"#### External Web Findings (`web_researcher`)\n{web_data['web_markdown']}")

        citations_raw = retrieval_data.get("citations_raw") or []
        bibliography = _format_citations_bibliography(citations_raw)
        final_answer = "\n\n".join(sections) + bibliography

        # Stage 3: Compliance & Faithfulness Auditor Subagent
        t_aud = time.perf_counter()
        contexts = [c.get("parent_content", "") for c in retrieval_data.get("relevant_chunks") or []]
        if sql_data is not None:
            contexts.append(sql_data["sql_markdown"])
        judge_res = rag_judge.evaluate_response(
            query=clean_query,
            answer=final_answer,
            contexts=contexts,
            use_llm_judge=False,
        )
        tax_res = execute_structured_taxonomy_query(
            StructuredQueryRequest(
                text=clean_query,
                allowed_labels=req.allowed_labels or DEFAULT_TAXONOMY_LABELS,
                retrieval_mode=req.retrieval_mode,
                top_k=req.top_k,
                tenant_id=req.tenant_id,
                user_roles=req.user_roles,
            )
        )
        aud_latency = round((time.perf_counter() - t_aud) * 1000.0, 2)
        compliance_audit = {
            **judge_res,
            "guardrail_flags": guardrail_flags,
            "taxonomy_label": tax_res.classification_label,
            "taxonomy_confidence": tax_res.confidence,
        }
        subagent_results.append(
            SubagentResult(
                agent_id="compliance_auditor",
                role="Compliance, Guardrails & Faithfulness Auditor",
                status="completed",
                latency_ms=aud_latency,
                tools_used=["sanitize_text", "classify_structured_taxonomy", "rag_judge_evaluate"],
                summary=(
                    f"Verdict={judge_res['verdict'].upper()} (faithfulness={judge_res['faithfulness_score']:.2f}, "
                    f"taxonomy={tax_res.classification_label})"
                ),
                output_preview=f"guardrail_flags={guardrail_flags}, hallucinations={judge_res['hallucinated_references']}",
            )
        )

        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        # Stage 4: Record distributed trace spans + tamper-evident audit ledger entry
        trace_id, waterfall_md = trace_collector.record_multi_agent_trace(
            query=clean_query,
            answer_preview=final_answer[:280],
            tenant_id=req.tenant_id,
            user_roles=req.user_roles,
            retrieval_mode=req.retrieval_mode,
            subagent_spans=[
                {
                    "name": f"subagent.{sr.agent_id}",
                    "duration_ms": sr.latency_ms,
                    "attributes": {"status": sr.status, "tools": ",".join(sr.tools_used)},
                }
                for sr in subagent_results
            ],
            faithfulness_score=judge_res["faithfulness_score"],
            citation_count=len(citations_raw),
            guardrail_flags=guardrail_flags,
            total_latency_ms=total_ms,
        )

        audit_entry = audit_logger.log_event(
            event_type="supervisor_multi_agent_orchestration",
            actor_tenant=req.tenant_id,
            actor_roles=req.user_roles,
            doc_ids=sorted({c["doc_id"] for c in citations_raw if c.get("doc_id")}),
            chunk_ids=[c["chunk_id"] for c in citations_raw if c.get("chunk_id")],
            summary=(
                f"Supervisor orchestrated {len(subagent_results)} subagents "
                f"({', '.join(s.agent_id for s in subagent_results)}) in {total_ms}ms"
            ),
            metadata={
                "trace_id": trace_id,
                "taxonomy_label": tax_res.classification_label,
                "judge_verdict": judge_res["verdict"],
            },
        )

        return OrchestrateResponse(
            query=clean_query,
            final_answer=final_answer,
            subagent_results=subagent_results,
            citations=[CitationItem(**c) for c in citations_raw],
            compliance_audit=compliance_audit,
            trace_id=trace_id,
            waterfall_markdown=waterfall_md,
            total_latency_ms=total_ms,
            audit_event_hash=audit_entry["event_hash"],
        )


supervisor_orchestrator = SupervisorOrchestrator()
