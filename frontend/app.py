"""Gradio Observability & Evaluation Studio for the Staff RAG Engineering Platform."""

import os
from typing import Any, Dict, Optional, Tuple
import gradio as gr
import requests


BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# --- Backend API Client Functions ---


def ingest_document_to_backend(
    content: str,
    title: str = "Untitled Document",
    source: str = "gradio_ui",
    chunking_strategy: str = "parent_child",
) -> Dict[str, Any]:
    """Sends a document to POST /ingest with chunking strategy and metadata."""
    payload = {
        "content": content,
        "title": title or "Untitled Document",
        "source": source or "gradio_ui",
        "chunking_strategy": chunking_strategy,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/ingest", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        return {"status": "error", "message": f"Connection error: {exc}"}


def seed_sample_corpus_on_backend(strategy: str = "parent_child") -> Dict[str, Any]:
    """Triggers POST /ingest-sample to load the built-in enterprise architecture docs."""
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
    retrieval_mode: str = "hybrid_rerank",
    top_k: int = 5,
    enable_query_expansion: bool = True,
    enable_web_search: bool = True,
    enable_self_rag: bool = True,
) -> Dict[str, Any]:
    """Submits an async RAG query job to the RabbitMQ queue via POST /submit-job."""
    payload = {
        "text": query,
        "retrieval_mode": retrieval_mode,
        "top_k": int(top_k),
        "enable_query_expansion": bool(enable_query_expansion),
        "enable_web_search": bool(enable_web_search),
        "enable_self_rag": bool(enable_self_rag),
    }
    try:
        response = requests.post(f"{BACKEND_URL}/submit-job", json=payload, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def get_job_result(job_id: str) -> Dict[str, Any]:
    """Polls GET /get-job/{job_id} for job status, answer, and pipeline trace."""
    try:
        response = requests.get(f"{BACKEND_URL}/get-job/{job_id}", timeout=15)
        if response.status_code == 404:
            return {"status": "not_found"}
        return response.json()
    except Exception as exc:
        return {"status": "error", "result": str(exc)}


def run_evaluation_on_backend(top_k: int = 3) -> Dict[str, Any]:
    """Calls POST /evaluate to run the Golden QA Ablation Study."""
    payload = {
        "top_k": int(top_k),
        "modes": ["dense", "sparse", "hybrid_rrf", "hybrid_rerank"],
        "seed_golden_corpus": True,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/evaluate", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# --- UI Formatting Helpers ---


def format_pipeline_trace_markdown(trace: Optional[Dict[str, Any]], tools: list) -> str:
    """Renders the PipelineTrace telemetry and retrieved chunk score table as Markdown."""
    if not trace:
        return "_No trace telemetry available yet._"

    latencies = trace.get("stage_latencies_ms") or {}
    latency_badges = " · ".join(f"`{k}: {v} ms`" for k, v in latencies.items())

    lines = [
        "### 🔬 Pipeline Observability & Retrieval Telemetry",
        f"- **Retrieval Mode:** `{trace.get('retrieval_mode')}` | "
        f"**CRAG Gate Decision:** `{trace.get('crag_decision')}` | "
        f"**Self-RAG Verdict:** `{trace.get('self_rag_verdict')}`",
        f"- **Faithfulness Score:** `{trace.get('faithfulness_score', 0.0):.3f}` | "
        f"**Inline Citations:** `{trace.get('citation_count', 0)}` | "
        f"**Total Latency:** `{trace.get('total_latency_ms', 0.0):.2f} ms`",
        f"- **Tools / Nodes Executed:** `{', '.join(tools) if tools else 'None'}`",
        f"- **Expanded Sub-Queries:** `{trace.get('expanded_queries', [])}`",
        f"- **Stage Latency Breakdown:** {latency_badges or 'N/A'}",
        "",
        "#### 📄 Two-Stage Hybrid Retrieval & Cross-Encoder Reranking Table",
        "| Rank | Chunk ID | Document Title | Dense Rank (Cosine) | BM25 Rank (Score) | RRF Score | Cross-Encoder Prob | CRAG Gate |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    chunks = trace.get("retrieved_chunks") or []
    if not chunks:
        lines.append("| - | None | _No chunks in corpus_ | - | - | - | - | - |")
    else:
        for idx, chk in enumerate(chunks, start=1):
            d_str = f"#{chk['dense_rank']} ({chk['dense_score']})" if chk.get("dense_rank") else "—"
            s_str = f"#{chk['sparse_rank']} ({chk['sparse_score']})" if chk.get("sparse_rank") else "—"
            gate_badge = "✅ PASS" if chk.get("is_relevant") else "❌ PRUNED"
            lines.append(
                f"| {idx} | `{chk.get('chunk_id', '')[:12]}` | {chk.get('title', '')} | "
                f"{d_str} | {s_str} | `{chk.get('rrf_score', 0.0):.5f}` | "
                f"**`{chk.get('calibrated_prob', 0.0):.3f}`** | {gate_badge} |"
            )

    return "\n".join(lines)


# --- Gradio Event Handlers ---


def ingest_document(
    content: str,
    title: str = "Untitled Document",
    source: str = "gradio_ui",
    strategy: str = "parent_child",
) -> str:
    if not content or not content.strip():
        return "⚠️ Please enter document content or click 'Seed Enterprise Sample Corpus'."
    response = ingest_document_to_backend(content, title=title, source=source, chunking_strategy=strategy)
    if response.get("status") == "error":
        return f"❌ Failed: {response['message']}"

    return (
        f"✅ **Ingested:** `{response.get('title')}` (`{response.get('doc_id')}`)\n"
        f"- **Strategy:** `{response.get('chunking_strategy')}`\n"
        f"- **Child Chunks Indexed (New):** `{response.get('chunks_indexed_new')}` "
        f"(Duplicates Skipped via SHA-256: `{response.get('chunks_skipped_duplicate')}`)\n"
        f"- **Parent Context Windows:** `{response.get('parent_windows_count')}`\n"
        f"- **Ingestion Latency:** `{response.get('ingestion_latency_ms')} ms`\n"
        f"- **Total Corpus Size:** `{response.get('corpus_stats', {}).get('total_chunks', 0)} chunks` across "
        f"`{response.get('corpus_stats', {}).get('total_documents', 0)} documents`"
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
    retrieval_mode: str = "hybrid_rerank",
    top_k: int = 5,
    enable_expansion: bool = True,
    enable_web: bool = True,
    enable_self_rag: bool = True,
) -> Tuple[str, Optional[str]]:
    if not query or not query.strip():
        return "⚠️ Please enter a query.", None

    response = submit_job_to_backend(
        query=query,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        enable_query_expansion=enable_expansion,
        enable_web_search=enable_web,
        enable_self_rag=enable_self_rag,
    )
    if "error" in response:
        return f"❌ Error: {response['error']}", None

    job_id = response.get("job_id")
    broker = response.get("broker", "rabbitmq")
    return f"✅ Job Submitted! ID: `{job_id}` (Broker: `{broker}`)\nClick **Check Job Result** to inspect output.", job_id


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
        return f"⚙️ Job `{job_id}` is **PROCESSING** through the LangGraph CRAG + Self-RAG pipeline...", "_Executing nodes..._"
    elif status == "completed":
        tool_Summary = f"\n\n🛠️ **Pipeline Tools Used:** `{', '.join(tools)}`" if tools else ""
        trace_md = format_pipeline_trace_markdown(trace, tools)
        return f"✅ **Grounded Response:**\n\n{result}{tool_Summary}", trace_md
    elif status == "failed":
        return f"❌ **Job Failed:** {result}", "_Pipeline execution failed._"
    else:
        return "Unknown job status.", "_No trace available._"


def handle_run_evaluation(top_k: int = 3) -> str:
    res = run_evaluation_on_backend(top_k=int(top_k))
    if res.get("status") == "error":
        return f"❌ Evaluation failed: {res.get('message')}"
    return res.get("markdown_report", "No report returned.")


# --- Gradio 3-Tab Studio Layout ---

with gr.Blocks(title="Staff RAG Engineering Studio") as demo:
    gr.Markdown(
        "# 🏗️ Staff RAG Engineering & Observability Studio\n"
        "**Two-Stage Hybrid Retrieval (`Milvus Dense` + `BM25` via `RRF k=60` $\\rightarrow$ `Cross-Encoder Reranker`) · "
        "LangGraph Corrective RAG (`CRAG`) & `Self-RAG` · Asynchronous `RabbitMQ` Workers**"
    )

    with gr.Tabs():
        # TAB 1: Ingestion & Chunking
        with gr.TabItem("1. Ingestion & Parent-Child Chunking"):
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
                    chunk_strategy = gr.Dropdown(
                        choices=["parent_child", "contextual_recursive", "fixed"],
                        value="parent_child",
                        label="Chunking Strategy (Small-to-Big Parent-Child vs. Contextual vs. Fixed Baseline)",
                    )
                    doc_input = gr.Textbox(
                        lines=8,
                        label="Document Content (Markdown / Technical Text)",
                        placeholder="Paste document text here or click 'Seed Enterprise Sample Corpus' on the right...",
                    )
                    with gr.Row():
                        ingest_btn = gr.Button("📥 Ingest Document", variant="primary")
                        seed_btn = gr.Button("🚀 Seed Enterprise Sample Corpus (3 Docs)")

                with gr.Column(scale=1):
                    ingest_out = gr.Markdown(
                        label="Ingestion & SHA-256 Deduplication Telemetry",
                        value="_Ready to ingest documents._",
                    )

        # TAB 2: Async Query & Trace Observability
        with gr.TabItem("2. Corrective & Self-RAG Query Studio"):
            with gr.Row():
                with gr.Column(scale=1):
                    q_input = gr.Textbox(
                        lines=3,
                        label="Submit Query",
                        value="What triggers ERR_LEDGER_9042 and what is the remediation command?",
                    )
                    retrieval_mode = gr.Dropdown(
                        choices=["hybrid_rerank", "hybrid_rrf", "dense", "sparse"],
                        value="hybrid_rerank",
                        label="Retrieval Pipeline Mode",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Top-K Chunks Passed to Synthesizer",
                    )
                    enable_expansion_chk = gr.Checkbox(value=True, label="Enable Multi-Query Expansion")
                    enable_web_chk = gr.Checkbox(value=True, label="Enable External Web Search (DuckDuckGo)")
                    enable_self_rag_chk = gr.Checkbox(value=True, label="Enable Self-RAG Faithfulness Grader")

                    submit_btn = gr.Button("📤 Submit Job to RabbitMQ Queue", variant="primary")
                    job_id_state = gr.State()
                    submission_out = gr.Markdown(value="_No active job submitted._")
                    check_btn = gr.Button("🔄 Check Job Result & Trace")

                with gr.Column(scale=2):
                    final_output = gr.Markdown(
                        label="Grounded & Cited Response",
                        value="_Submit a query and click 'Check Job Result' to view the cited response._",
                    )
                    trace_output = gr.Markdown(
                        label="Pipeline Trace & Retrieval Score Breakdown",
                        value="_Retrieval telemetry will appear here._",
                    )

        # TAB 3: Quantitative RAG Evaluation & Ablation Suite
        with gr.TabItem("3. RAG Evaluation & Ablation Benchmark"):
            gr.Markdown(
                "### 📈 Quantitative IR & RAG Ablation Suite\n"
                "Benchmarks **Dense (`all-MiniLM-L6-v2`)**, **Sparse (`BM25Okapi`)**, **Hybrid RRF (`k=60`)**, "
                "and **Hybrid + Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`)** across the built-in "
                "Golden Technical QA Dataset (`HitRate@K`, `MRR@K`, `NDCG@K`, `ContextPrecision@K`, `Latency ms`)."
            )
            with gr.Row():
                eval_top_k = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Evaluation Cut-off (Top-K)")
                eval_btn = gr.Button("🧪 Run Full Retrieval Ablation Study", variant="primary")
            eval_output = gr.Markdown(value="_Click 'Run Full Retrieval Ablation Study' to compute metrics._")

    # Event Wiring
    ingest_btn.click(
        ingest_document,
        inputs=[doc_input, doc_title, doc_source, chunk_strategy],
        outputs=ingest_out,
    )
    seed_btn.click(
        handle_seed_sample_corpus,
        inputs=[chunk_strategy],
        outputs=ingest_out,
    )
    submit_btn.click(
        submit_query,
        inputs=[
            q_input,
            retrieval_mode,
            top_k_slider,
            enable_expansion_chk,
            enable_web_chk,
            enable_self_rag_chk,
        ],
        outputs=[submission_out, job_id_state],
    )
    check_btn.click(
        check_status,
        inputs=job_id_state,
        outputs=[final_output, trace_output],
    )
    eval_btn.click(
        handle_run_evaluation,
        inputs=[eval_top_k],
        outputs=eval_output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
