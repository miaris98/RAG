"""Gradio Observability & Evaluation Studio for the Staff+ RAG Engineering Platform."""

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple
import gradio as gr
import requests


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# --- Backend API Client Functions ---


def ingest_document_to_backend(
    content: str,
    title: str = "Untitled Document",
    source: str = "gradio_ui",
    chunking_strategy: str = "parent_child",
    tenant_id: str = "default",
    access_roles: Optional[List[str]] = None,
    apply_guardrails: bool = True,
    enable_contextual_enrichment: bool = False,
) -> Dict[str, Any]:
    """Sends a document to POST /ingest with RBAC metadata, guardrails, and chunking strategy."""
    payload = {
        "content": content,
        "title": title or "Untitled Document",
        "source": source or "gradio_ui",
        "chunking_strategy": chunking_strategy,
        "tenant_id": tenant_id or "default",
        "access_roles": access_roles or ["public"],
        "apply_guardrails": bool(apply_guardrails),
        "enable_contextual_enrichment": bool(enable_contextual_enrichment),
    }
    try:
        response = requests.post(f"{BACKEND_URL}/ingest", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        return {"status": "error", "message": f"Connection error: {exc}"}


def seed_sample_corpus_on_backend(strategy: str = "parent_child") -> Dict[str, Any]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/ingest-sample",
            params={"strategy": strategy},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        return {"status": "error", "message": f"Connection error: {exc}"}


def submit_job_to_backend(
    query: str,
    retrieval_mode: str = "hybrid_graph_rerank",
    top_k: int = 5,
    tenant_id: str = "default",
    user_roles: Optional[List[str]] = None,
    enable_query_expansion: bool = True,
    enable_web_search: bool = True,
    enable_self_rag: bool = True,
    enable_semantic_cache: bool = True,
    enable_guardrails: bool = True,
) -> Dict[str, Any]:
    payload = {
        "text": query,
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k),
        "tenant_id": tenant_id or "default",
        "user_roles": user_roles or ["public"],
        "enable_query_expansion": bool(enable_query_expansion),
        "enable_web_search": bool(enable_web_search),
        "enable_self_rag": bool(enable_self_rag),
        "enable_semantic_cache": bool(enable_semantic_cache),
        "enable_guardrails": bool(enable_guardrails),
    }
    try:
        response = requests.post(f"{BACKEND_URL}/submit-job", json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def get_job_result(job_id: str) -> Dict[str, Any]:
    try:
        response = requests.get(f"{BACKEND_URL}/get-job/{job_id}", timeout=15)
        if response.status_code == 404:
            return {"status": "not_found"}
        return response.json()
    except Exception as exc:
        return {"status": "error", "result": str(exc)}


def run_evaluation_on_backend(top_k: int = 3, enable_judge_eval: bool = True) -> Dict[str, Any]:
    payload = {
        "top_k": int(top_k),
        "modes": ["dense", "sparse", "hybrid_rrf", "hybrid_rerank", "hybrid_graph_rerank"],
        "seed_golden_corpus": True,
        "enable_judge_eval": bool(enable_judge_eval),
        "use_llm_judge": False,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/evaluate", json=payload, timeout=90)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def fetch_compliance_audit_dashboard() -> Tuple[str, str, str, str]:
    """Fetches /compliance/audit-report and /metrics/prometheus and formats all 4 Audit Center panels."""
    try:
        report_resp = requests.get(f"{BACKEND_URL}/compliance/audit-report", params={"limit": 50}, timeout=15)
        report_resp.raise_for_status()
        report = report_resp.json()
    except Exception as exc:
        err = f"❌ Could not fetch audit report from backend: `{exc}`"
        return err, "_Unavailable_", "_Unavailable_", "# Error fetching Prometheus metrics"

    try:
        prom_resp = requests.get(f"{BACKEND_URL}/metrics/prometheus", timeout=10)
        prom_resp.raise_for_status()
        prom_text = prom_resp.text
    except Exception as exc:
        prom_text = f"# Error fetching /metrics/prometheus: {exc}"

    dossier_md = report.get("markdown_dossier", "_No dossier available._")

    # Format Distributed Traces (Langfuse / Span Waterfall)
    traces = report.get("recent_traces") or []
    trace_lines = [
        "### 🔭 Distributed Traces & Span Waterfall (Langfuse / OpenTelemetry)",
        "| Trace ID | Query (PII-Sanitized) | Route | Cache | Faithfulness | Latency | Node Spans | Langfuse |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
    ]
    if not traces:
        trace_lines.append("| - | _No queries traced yet_ | - | - | - | - | - | - |")
    else:
        for tr in traces:
            spans_str = " → ".join(f"`{s['name']}:{s['duration_ms']}ms`" for s in tr.get("spans", []))
            q_short = (tr.get("query", "")[:42] + "...") if len(tr.get("query", "")) > 42 else tr.get("query", "")
            trace_lines.append(
                f"| `{tr.get('trace_id', '')[:10]}` | {q_short} | `{tr.get('query_route')}` | "
                f"`{tr.get('cache_hit')}` | `{tr.get('faithfulness_score', 0.0):.2f}` | "
                f"`{tr.get('total_latency_ms', 0.0):.2f} ms` | {spans_str} | `{tr.get('langfuse_status')}` |"
            )
        latest_waterfall = traces[0].get("waterfall_markdown")
        if latest_waterfall:
            trace_lines.extend(
                [
                    "",
                    f"#### ⏱️ Latest Trace Hierarchical Span Waterfall (`{traces[0].get('trace_id')}`)",
                    latest_waterfall,
                ]
            )

    # Format SHA-256 Tamper-Evident Audit Ledger
    events = report.get("audit_events") or []
    chain_badge = "✅ CHAIN VALID" if report.get("chain_valid") else "❌ TAMPER DETECTED"
    ledger_lines = [
        f"### 🔐 Tamper-Evident SHA-256 Audit Ledger ({chain_badge})",
        "| Seq | Event Type | Tenant | Roles | Doc IDs | Summary | Hash Chain (`prev -> event`) |",
        "| :---: | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    if not events:
        ledger_lines.append("| - | _No audit events logged yet_ | - | - | - | - | - |")
    else:
        for ev in events:
            docs_str = ", ".join(f"`{d}`" for d in ev.get("doc_ids", [])) or "—"
            roles_str = ", ".join(ev.get("actor_roles", [])) or "—"
            h_str = f"`{ev.get('prev_hash', '')[:8]}→{ev.get('event_hash', '')[:8]}`"
            ledger_lines.append(
                f"| {ev.get('seq')} | `{ev.get('event_type')}` | `{ev.get('actor_tenant')}` | "
                f"`{roles_str}` | {docs_str} | {ev.get('summary', '')} | {h_str} |"
            )

    return dossier_md, "\n".join(trace_lines), "\n".join(ledger_lines), prom_text


def handle_gdpr_erasure(doc_id: str) -> Tuple[str, str, str, str, str]:
    """Executes GDPR Article 17 deletion via DELETE /documents/{doc_id} and refreshes the Audit Center."""
    cleaned_id = (doc_id or "").strip()
    if not cleaned_id:
        dossier, traces, ledger, prom = fetch_compliance_audit_dashboard()
        return "⚠️ Please enter a `doc_id` from the Active Corpus table above.", dossier, traces, ledger, prom

    try:
        resp = requests.delete(f"{BACKEND_URL}/documents/{cleaned_id}", timeout=15)
        resp.raise_for_status()
        res = resp.json()
        status_msg = (
            f"✅ **GDPR Article 17 Erasure Complete:** `{res.get('doc_id')}` "
            f"(`chunks_deleted={res.get('chunks_deleted')}`, `cache_evicted={res.get('cache_entries_evicted')}`, "
            f"`audit_hash={str(res.get('audit_event_hash', ''))[:12]}...`)"
        )
    except Exception as exc:
        status_msg = f"❌ Erasure failed: {exc}"

    dossier, traces, ledger, prom = fetch_compliance_audit_dashboard()
    return status_msg, dossier, traces, ledger, prom


# --- UI Formatting Helpers ---


def format_pipeline_trace_markdown(trace: Optional[Dict[str, Any]], tools: list) -> str:
    if not trace:
        return "_No trace telemetry available yet._"

    latencies = trace.get("stage_latencies_ms") or {}
    latency_badges = " · ".join(f"`{k}: {v} ms`" for k, v in latencies.items())
    guard_flags = trace.get("guardrail_flags") or []
    funnel = trace.get("retrieval_funnel") or {}

    lines = [
        "### 🔬 End-to-End Distributed Trace & Retrieval Visibility",
        f"- **Distributed Trace ID:** `{trace.get('trace_id', 'N/A')}` | **Query Route:** `{trace.get('query_route')}` | "
        f"**Retrieval Mode:** `{trace.get('retrieval_mode')}` | **Semantic Cache Hit:** `{trace.get('cache_hit')}`",
        f"- **RBAC Scope:** `tenant={trace.get('tenant_id')}`, `roles={trace.get('user_roles')}` | "
        f"**Guardrail Flags:** `{guard_flags or ['clean']}`",
        f"- **CRAG Gate:** `{trace.get('crag_decision')}` | **Self-RAG Verdict:** `{trace.get('self_rag_verdict')}` | "
        f"**Faithfulness:** `{trace.get('faithfulness_score', 0.0):.3f}` | **Inline Citations:** `{trace.get('citation_count', 0)}`",
        f"- **Total Latency:** `{trace.get('total_latency_ms', 0.0):.2f} ms` | **Tools:** `{', '.join(tools) if tools else 'None'}`",
        f"- **Expanded Sub-Queries:** `{trace.get('expanded_queries', [])}`",
        f"- **Stage Latency Breakdown:** {latency_badges or 'N/A'}",
    ]

    if funnel:
        lines.extend(
            [
                "",
                "#### 🔎 8-Stage Retrieval Candidate Funnel",
                f"`Corpus: {funnel.get('total_corpus_chunks', 0)}` → "
                f"`RBAC Authorized: {funnel.get('rbac_authorized_chunks', 0)}` → "
                f"`Dense/BM25/Graph: {funnel.get('dense_candidates_count', 0)}/{funnel.get('sparse_candidates_count', 0)}/{funnel.get('graph_candidates_count', 0)}` → "
                f"`RRF Pool: {funnel.get('rrf_fused_pool_count', 0)}` → "
                f"`Parent Windows: {funnel.get('parent_promoted_count', 0)}` → "
                f"`CRAG Relevant: {funnel.get('crag_relevant_count', 0)}` → "
                f"`MMR Selected: {funnel.get('mmr_selected_count', 0)}`",
            ]
        )

    lines.extend(
        [
            "",
            "#### 📄 Tri-Modal Retrieval Explainability & Prompt Positioning",
            "| Rank | Prompt Slot | Chunk ID | Title | RRF Breakdown (D / S / G) | Matched Terms | Matched Entities | Cross-Encoder Prob | CRAG |",
            "| :---: | :---: | :--- | :--- | :---: | :--- | :--- | :---: | :---: |",
        ]
    )

    chunks = trace.get("retrieved_chunks") or []
    if not chunks:
        lines.append("| - | — | None | _No authorized chunks matched_ | — | — | — | — | — |")
    else:
        for idx, chk in enumerate(chunks, start=1):
            slot = f"#{chk['prompt_position']}" if chk.get("prompt_position") else "—"
            rb = chk.get("rrf_breakdown") or {}
            rrf_str = f"`{rb.get('dense', 0):.4f}` / `{rb.get('sparse', 0):.4f}` / `{rb.get('graph', 0):.4f}`"
            terms_str = ", ".join(f"`{t}`" for t in (chk.get("matched_terms") or [])[:4]) or "—"
            ents_str = ", ".join(f"`{e}`" for e in (chk.get("matched_entities") or [])[:3]) or "—"
            gate_badge = "✅ PASS" if chk.get("is_relevant") else "❌ PRUNED"
            lines.append(
                f"| {idx} | {slot} | `{chk.get('chunk_id', '')[:12]}` | {chk.get('title', '')} | "
                f"{rrf_str} | {terms_str} | {ents_str} | **`{chk.get('calibrated_prob', 0.0):.3f}`** | {gate_badge} |"
            )

    pruned = funnel.get("pruned_chunks") or []
    if pruned:
        pruned_str = ", ".join(f"`{p['chunk_id'][:10]}` ({p['reason']}, p={p['calibrated_prob']:.2f})" for p in pruned)
        lines.append(f"\n- **Pruned / Filtered Candidates:** {pruned_str}")

    waterfall = trace.get("waterfall_markdown")
    if waterfall:
        lines.extend(["", "#### ⏱️ Distributed Trace Span Waterfall", waterfall])

    citations = trace.get("citations") or []
    if citations:
        lines.extend(["", "#### 📚 Structured Inline Citations Extracted"])
        for cit in citations:
            lines.append(
                f"- **{cit['ref_id']}** `{cit['title']}` (§{cit['section']}, `{cit['chunk_id'][:12]}`, "
                f"prob=`{cit['calibrated_prob']:.3f}`): _{cit['supporting_quote']}_"
            )

    return "\n".join(lines)


# --- Gradio Event Handlers ---


def ingest_document(
    content: str,
    title: str,
    source: str,
    strategy: str,
    tenant_id: str,
    roles_selected: List[str],
    apply_guardrails: bool,
    enable_contextual: bool = False,
) -> str:
    if not content or not content.strip():
        return "⚠️ Please enter document content or click 'Seed Enterprise Sample Corpus'."
    response = ingest_document_to_backend(
        content=content,
        title=title,
        source=source,
        chunking_strategy=strategy,
        tenant_id=tenant_id,
        access_roles=roles_selected or ["public"],
        apply_guardrails=apply_guardrails,
        enable_contextual_enrichment=enable_contextual,
    )
    if response.get("status") == "error":
        return f"❌ Failed: {response['message']}"

    symbols = response.get("code_symbols_extracted") or []
    sym_str = ", ".join(f"`{s}`" for s in symbols[:10]) if symbols else "`none`"
    return (
        f"✅ **Ingested:** `{response.get('title')}` (`{response.get('doc_id')}`)\n"
        f"- **RBAC Scope:** `tenant={response.get('tenant_id')}`, `roles={response.get('access_roles')}`\n"
        f"- **Guardrail Flags:** `{response.get('guardrail_flags') or ['clean']}` | "
        f"**Contextual Enrichment:** `{response.get('contextual_enrichment_enabled', False)}` | "
        f"**Cache Entries Invalidated:** `{response.get('invalidated_cache_entries', 0)}`\n"
        f"- **Code Symbols Extracted (AST/Polyglot):** {sym_str}\n"
        f"- **Child Chunks Indexed (New):** `{response.get('chunks_indexed_new')}` "
        f"(Duplicates Skipped via SHA-256: `{response.get('chunks_skipped_duplicate')}`)\n"
        f"- **Latency:** `{response.get('ingestion_latency_ms')} ms`"
    )


def handle_seed_sample_corpus(strategy: str = "parent_child") -> str:
    response = seed_sample_corpus_on_backend(strategy=strategy)
    if response.get("status") == "error":
        return f"❌ Failed: {response['message']}"
    docs = ", ".join(response.get("documents_loaded", []))
    return (
        f"✅ **Enterprise Sample Corpus Indexed!**\n"
        f"- **Documents:** {docs}\n"
        f"- **New Chunks Indexed:** `{response.get('chunks_indexed_new')}` "
        f"(Duplicates Skipped: `{response.get('chunks_skipped_duplicate')}`)\n"
        f"- **Latency:** `{response.get('ingestion_latency_ms')} ms`"
    )


def submit_query(
    query: str,
    retrieval_mode: str,
    top_k: int,
    tenant_id: str,
    user_roles: List[str],
    enable_expansion: bool,
    enable_web: bool,
    enable_self_rag: bool,
    enable_cache: bool,
    enable_guardrails: bool,
) -> Tuple[str, Optional[str]]:
    if not query or not query.strip():
        return "⚠️ Please enter a query.", None

    response = submit_job_to_backend(
        query=query,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        tenant_id=tenant_id,
        user_roles=user_roles or ["public"],
        enable_query_expansion=enable_expansion,
        enable_web_search=enable_web,
        enable_self_rag=enable_self_rag,
        enable_semantic_cache=enable_cache,
        enable_guardrails=enable_guardrails,
    )
    if "error" in response:
        return f"❌ Error: {response['error']}", None

    job_id = response.get("job_id")
    broker = response.get("broker", "rabbitmq")
    return f"✅ Job Submitted! ID: `{job_id}` (Broker: `{broker}`)\nClick **Check Job Result**.", job_id


def check_status(job_id: Optional[str]) -> Tuple[str, str]:
    if not job_id:
        return "No active job. Submit a query first.", "_No trace telemetry available yet._"

    status_response = get_job_result(job_id)
    status = status_response.get("status")
    result = status_response.get("result")
    tools = status_response.get("tools", [])
    trace = status_response.get("trace")

    if status == "queued":
        return f"⏳ Job `{job_id}` is **QUEUED**.", "_Waiting for worker..._"
    elif status == "processing":
        return f"⚙️ Job `{job_id}` is **PROCESSING**...", "_Executing LangGraph nodes..._"
    elif status == "completed":
        trace_md = format_pipeline_trace_markdown(trace, tools)
        return f"✅ **Grounded Response:**\n\n{result}", trace_md
    elif status == "failed":
        return f"❌ **Job Failed:** {result}", "_Pipeline execution failed._"
    else:
        return "Unknown job status.", "_No trace available._"


def stream_query_live(
    query: str,
    retrieval_mode: str,
    top_k: int,
    tenant_id: str,
    user_roles: List[str],
    enable_expansion: bool,
    enable_web: bool,
    enable_self_rag: bool,
    enable_cache: bool,
    enable_guardrails: bool,
) -> Iterator[Tuple[str, str]]:
    """Consumes POST /query-stream SSE events and yields live updates to Gradio."""
    if not query or not query.strip():
        yield "⚠️ Please enter a query.", "_Waiting for query..._"
        return

    payload = {
        "text": query,
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k),
        "tenant_id": tenant_id or "default",
        "user_roles": user_roles or ["public"],
        "enable_query_expansion": bool(enable_expansion),
        "enable_web_search": bool(enable_web),
        "enable_self_rag": bool(enable_self_rag),
        "enable_semantic_cache": bool(enable_cache),
        "enable_guardrails": bool(enable_guardrails),
    }

    progress_lines: List[str] = ["⚡ **Live LangGraph SSE Stream Started:**"]
    try:
        with requests.post(f"{BACKEND_URL}/query-stream", json=payload, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                evt = json.loads(raw_line[len("data: ") :])
                if evt.get("event") == "stage":
                    progress_lines.append(f"- Node `{evt.get('node')}` → `{evt.get('status')}`")
                    yield "\n".join(progress_lines), "_Streaming LangGraph execution..._"
                elif evt.get("event") == "final":
                    trace_md = format_pipeline_trace_markdown(evt.get("trace"), evt.get("tools", []))
                    yield f"✅ **Grounded Response (SSE Streamed):**\n\n{evt.get('answer', '')}", trace_md
    except Exception as exc:
        yield f"❌ SSE Stream Error: {exc}", "_Stream interrupted._"


def handle_run_evaluation(top_k: int = 3, enable_judge_eval: bool = True) -> str:
    res = run_evaluation_on_backend(top_k=int(top_k), enable_judge_eval=bool(enable_judge_eval))
    if res.get("status") == "error":
        return f"❌ Evaluation failed: {res.get('message')}"
    return res.get("markdown_report", "No report returned.")


# --- Gradio 3-Tab Studio Layout ---

with gr.Blocks(title="Staff+ RAG Engineering Studio") as demo:
    gr.Markdown(
        "# 🏗️ Staff+ RAG Engineering & Observability Studio\n"
        "**Pre-Filtered RBAC · Tri-Modal GraphRAG (`Milvus Dense` + `BM25` + `Entity Graph` via `RRF k=60` $\\rightarrow$ `Cross-Encoder`) · "
        "Adaptive Query Router · Semantic Cache · Guardrails · LangGraph `CRAG`/`Self-RAG` · SSE Streaming**"
    )

    with gr.Tabs():
        # TAB 1: Ingestion, RBAC & Guardrails
        with gr.TabItem("1. Ingestion, RBAC & Guardrails"):
            with gr.Row():
                with gr.Column(scale=2):
                    doc_title = gr.Textbox(
                        label="Document Title",
                        value="AegisPay Distributed Payment Ledger Architecture (RFC-402)",
                    )
                    doc_source = gr.Textbox(
                        label="Source Identifier / URI",
                        value="internal://architecture/rfc-402.md",
                    )
                    with gr.Row():
                        chunk_strategy = gr.Dropdown(
                            choices=["parent_child", "contextual_recursive", "fixed"],
                            value="parent_child",
                            label="Chunking Strategy",
                        )
                        ingest_tenant = gr.Textbox(label="Tenant ID", value="default")
                        ingest_roles = gr.CheckboxGroup(
                            choices=["public", "engineering", "compliance"],
                            value=["public", "engineering"],
                            label="Authorized RBAC Roles",
                        )
                    with gr.Row():
                        ingest_guardrails = gr.Checkbox(
                            value=True,
                            label="Enable Security Guardrails (PII Redaction & Prompt Injection Defense)",
                        )
                        ingest_contextual = gr.Checkbox(
                            value=True,
                            label="Enable Anthropic Contextual Enrichment (Document Summary Prefix)",
                        )
                    doc_input = gr.Textbox(
                        lines=7,
                        label="Document Content",
                        placeholder="Paste document text here or click 'Seed Enterprise Sample Corpus'...",
                    )
                    with gr.Row():
                        ingest_btn = gr.Button("📥 Ingest Document", variant="primary")
                        seed_btn = gr.Button("🚀 Seed Enterprise Sample Corpus (3 Docs)")

                with gr.Column(scale=1):
                    ingest_out = gr.Markdown(
                        label="Ingestion, RBAC & Guardrails Telemetry",
                        value="_Ready to ingest documents._",
                    )

        # TAB 2: Async / SSE Streaming Query Studio
        with gr.TabItem("2. Corrective & Self-RAG Query Studio"):
            with gr.Row():
                with gr.Column(scale=1):
                    q_input = gr.Textbox(
                        lines=3,
                        label="Submit Query",
                        value="What triggers ERR_LEDGER_9042 and what is the remediation command?",
                    )
                    retrieval_mode = gr.Dropdown(
                        choices=["hybrid_graph_rerank", "hybrid_rerank", "hybrid_rrf", "dense", "sparse"],
                        value="hybrid_graph_rerank",
                        label="Retrieval Pipeline Mode",
                    )
                    top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Top-K Chunks")
                    query_tenant = gr.Textbox(label="Caller Tenant ID", value="default")
                    query_roles = gr.CheckboxGroup(
                        choices=["public", "engineering", "compliance"],
                        value=["public", "engineering", "compliance"],
                        label="Caller RBAC Roles (Pre-Filtered)",
                    )
                    enable_expansion_chk = gr.Checkbox(value=True, label="Adaptive Query Router (Lexical / HyDE / Multi-Hop)")
                    enable_web_chk = gr.Checkbox(value=True, label="External Web Search (DuckDuckGo)")
                    enable_self_rag_chk = gr.Checkbox(value=True, label="Self-RAG Faithfulness Grader")
                    enable_cache_chk = gr.Checkbox(value=True, label="Role-Scoped Semantic Query Cache")
                    enable_guardrails_chk = gr.Checkbox(value=True, label="PII & Injection Guardrails")

                    with gr.Row():
                        stream_btn = gr.Button("⚡ Stream Live (SSE)", variant="primary")
                        submit_btn = gr.Button("📤 Queue via RabbitMQ")
                    job_id_state = gr.State()
                    submission_out = gr.Markdown(value="_No active job submitted._")
                    check_btn = gr.Button("🔄 Check RabbitMQ Job Result")

                with gr.Column(scale=2):
                    final_output = gr.Markdown(
                        label="Grounded & Cited Response",
                        value="_Stream or submit a query to view the cited response._",
                    )
                    trace_output = gr.Markdown(
                        label="Pipeline Trace & Tri-Modal Score Breakdown",
                        value="_Retrieval telemetry will appear here._",
                    )

        # TAB 3: Quantitative RAG Evaluation & Ablation Suite
        with gr.TabItem("3. RAG Evaluation & Ablation Benchmark"):
            gr.Markdown(
                "### 📈 Quantitative IR & RAG Ablation Suite\n"
                "Benchmarks **Dense**, **Sparse (BM25)**, **Hybrid RRF**, **Hybrid + Cross-Encoder**, "
                "and **Tri-Modal GraphRAG (Dense + BM25 + Entity Graph + Cross-Encoder)** across the Golden QA Dataset."
            )
            with gr.Row():
                eval_top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Evaluation Cut-off (Top-K)")
                eval_judge_chk = gr.Checkbox(
                    value=True,
                    label="Enable P2.1 LLM-as-a-Judge & Human-Calibration Audit",
                )
                eval_btn = gr.Button("🧪 Run Full 5-Mode Retrieval Ablation Study", variant="primary")
            eval_output = gr.Markdown(value="_Click 'Run Full 5-Mode Retrieval Ablation Study' to compute metrics._")

        # TAB 4: Compliance Audit & End-to-End Observability Center
        with gr.TabItem("4. 🛡️ Compliance Audit & Observability Center"):
            gr.Markdown(
                "### 🛡️ System Audit, GDPR Article 17 Erasure & Full-Stack Observability\n"
                "Inspect the **cryptographic SHA-256 audit chain**, **active document RBAC lineage**, "
                "**Langfuse-compatible distributed traces & spans**, and live **Prometheus** metrics "
                "(scraped by Grafana at `:3000` in Docker Compose)."
            )
            with gr.Row():
                audit_refresh_btn = gr.Button("🔄 Refresh Audit & Observability Dashboard", variant="primary")
                erase_doc_input = gr.Textbox(
                    label="GDPR Article 17 Erasure — Target doc_id",
                    placeholder="Paste a doc_id from the Active Corpus table (e.g. doc_...)",
                )
                erase_btn = gr.Button("🗑️ Erase Document (GDPR Art. 17)", variant="stop")
            erase_status_md = gr.Markdown(value="_Ready for compliance inspection or GDPR erasure._")

            audit_dossier_md = gr.Markdown(value="_Click 'Refresh Audit & Observability Dashboard' to load the compliance dossier._")
            audit_traces_md = gr.Markdown(value="_Distributed trace waterfall will appear here._")
            audit_ledger_md = gr.Markdown(value="_Tamper-evident SHA-256 audit ledger will appear here._")
            prom_metrics_code = gr.Code(
                label="Live Prometheus Exposition (/metrics/prometheus)",
                language="shell",
                value="# Click 'Refresh Audit & Observability Dashboard' to inspect live Prometheus metrics.",
            )

    # Event Wiring
    ingest_btn.click(
        ingest_document,
        inputs=[
            doc_input,
            doc_title,
            doc_source,
            chunk_strategy,
            ingest_tenant,
            ingest_roles,
            ingest_guardrails,
            ingest_contextual,
        ],
        outputs=ingest_out,
    )
    seed_btn.click(handle_seed_sample_corpus, inputs=[chunk_strategy], outputs=ingest_out)

    stream_btn.click(
        stream_query_live,
        inputs=[
            q_input, retrieval_mode, top_k_slider, query_tenant, query_roles,
            enable_expansion_chk, enable_web_chk, enable_self_rag_chk, enable_cache_chk, enable_guardrails_chk,
        ],
        outputs=[final_output, trace_output],
    )
    submit_btn.click(
        submit_query,
        inputs=[
            q_input, retrieval_mode, top_k_slider, query_tenant, query_roles,
            enable_expansion_chk, enable_web_chk, enable_self_rag_chk, enable_cache_chk, enable_guardrails_chk,
        ],
        outputs=[submission_out, job_id_state],
    )
    check_btn.click(check_status, inputs=job_id_state, outputs=[final_output, trace_output])
    eval_btn.click(handle_run_evaluation, inputs=[eval_top_k, eval_judge_chk], outputs=eval_output)
    audit_refresh_btn.click(
        fetch_compliance_audit_dashboard,
        inputs=[],
        outputs=[audit_dossier_md, audit_traces_md, audit_ledger_md, prom_metrics_code],
    )
    erase_btn.click(
        handle_gdpr_erasure,
        inputs=[erase_doc_input],
        outputs=[erase_status_md, audit_dossier_md, audit_traces_md, audit_ledger_md, prom_metrics_code],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
