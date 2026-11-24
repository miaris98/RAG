"""Automated RAG Ablation Benchmark Runner comparing:
1. Dense-Only (Bi-Encoder Cosine in Milvus)
2. Sparse-Only (BM25Okapi Lexical Index)
3. Hybrid RRF (Dense + BM25 Reciprocal Rank Fusion, k=60)
4. Hybrid RRF + Cross-Encoder Reranker (Two-Stage Architecture)
"""

import time
from typing import Any, Dict, List

from app.evaluation.judge import rag_judge
from app.evaluation.metrics import (
    compute_relevance_vector,
    context_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    reciprocal_rank_at_k,
)
from app.ingestion.pipeline import ingestion_pipeline
from app.models import EvaluationRequest, QueryInput, RetrievalMode
from app.retrieval.hybrid import retrieval_engine


GOLDEN_QA_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "Q1_exact_error_code",
        "category": "Exact Identifier / Error Code",
        "query": "What triggers ERR_LEDGER_9042 and what is the remediation command?",
        "expected_keywords": ["err_ledger_9042", "aegis-ctl raft resync"],
    },
    {
        "id": "Q2_exact_rotation_id",
        "category": "Exact Identifier / Contact",
        "query": "Which PagerDuty rotation ID handles sovereign regulatory inquiries in EMEA?",
        "expected_keywords": ["pd-emea-8821"],
    },
    {
        "id": "Q3_semantic_settlement",
        "category": "Semantic Paraphrase",
        "query": "How fast does AegisPay clear cross-border liquidity pools compared to traditional card rails?",
        "expected_keywords": ["180 milliseconds", "raft consensus"],
    },
    {
        "id": "Q4_rag_slo_threshold",
        "category": "Technical Architecture SLO",
        "query": "What is the p95 retrieval latency SLO and maximum unfaithfulness rate for production RAG?",
        "expected_keywords": ["85ms", "1.5%"],
    },
    {
        "id": "Q5_governance_president",
        "category": "Entity & Temporal Grounding",
        "query": "Who is the President of the Hellenic Republic according to the internal governance directory?",
        "expected_keywords": ["constantine tassoulas", "katerina sakellaropoulou"],
    },
    {
        "id": "Q6_encryption_retention",
        "category": "Security & Storage Policy",
        "query": "Which MinIO bucket and encryption cipher store transaction audit embeddings, and for how many days?",
        "expected_keywords": ["aes-256-gcm", "aegis-audit-prod", "90 days"],
    },
]


def run_ablation_benchmark(req: EvaluationRequest) -> Dict[str, Any]:
    """Executes the Golden QA Benchmark across requested retrieval modes and computes IR metrics."""
    if req.seed_golden_corpus:
        ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    mode_summaries: List[Dict[str, Any]] = []
    per_query_details: Dict[str, List[Dict[str, Any]]] = {}

    for mode in req.modes:
        hits: List[float] = []
        mrrs: List[float] = []
        ndcgs: List[float] = []
        precisions: List[float] = []
        latencies: List[float] = []
        query_rows: List[Dict[str, Any]] = []

        for item in GOLDEN_QA_BENCHMARK:
            t0 = time.perf_counter()
            retrieved = retrieval_engine.retrieve(
                queries=[item["query"]],
                mode=mode,  # type: ignore[arg-type]
                top_k=req.top_k,
                promote_parents=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            texts = [f"{c.content}\n{c.parent_content}" for c in retrieved]
            rel_vec = compute_relevance_vector(texts, item["expected_keywords"], k=req.top_k)

            hr = hit_rate_at_k(rel_vec)
            rr = reciprocal_rank_at_k(rel_vec)
            ndcg = ndcg_at_k(rel_vec)
            prec = context_precision_at_k(rel_vec)

            hits.append(hr)
            mrrs.append(rr)
            ndcgs.append(ndcg)
            precisions.append(prec)

            query_rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "query": item["query"],
                    "relevance_vector": rel_vec,
                    "hit_rate": round(hr, 4),
                    "mrr": round(rr, 4),
                    "ndcg": round(ndcg, 4),
                    "top_chunk_title": retrieved[0].title if retrieved else "None",
                    "top_calibrated_prob": retrieved[0].calibrated_prob if retrieved else 0.0,
                    "latency_ms": round(elapsed_ms, 2),
                }
            )

        n = max(1, len(GOLDEN_QA_BENCHMARK))
        summary = {
            "mode": mode,
            "top_k": req.top_k,
            "num_queries": len(GOLDEN_QA_BENCHMARK),
            "hit_rate_at_k": round(sum(hits) / n, 4),
            "mrr_at_k": round(sum(mrrs) / n, 4),
            "ndcg_at_k": round(sum(ndcgs) / n, 4),
            "context_precision_at_k": round(sum(precisions) / n, 4),
            "mean_latency_ms": round(sum(latencies) / n, 2),
        }
        mode_summaries.append(summary)
        per_query_details[mode] = query_rows

    markdown_report = _format_markdown_ablation_table(mode_summaries, req.top_k)
    result: Dict[str, Any] = {
        "status": "success",
        "top_k": req.top_k,
        "corpus_stats": retrieval_engine.stats(),
        "summaries": mode_summaries,
        "per_query_details": per_query_details,
        "markdown_report": markdown_report,
    }

    if req.enable_judge_eval:
        from app.agent.graph import execute_rag_pipeline

        judge_rows: List[Dict[str, Any]] = []
        for item in GOLDEN_QA_BENCHMARK:
            rag_out = execute_rag_pipeline(
                QueryInput(
                    text=item["query"],
                    retrieval_mode="hybrid_graph_rerank",
                    top_k=req.top_k,
                    enable_web_search=False,
                    enable_semantic_cache=False,
                    max_retries=0,
                )
            )
            chunks = rag_out.get("trace", {}).get("retrieved_chunks", [])
            contexts = [c.get("parent_content", "") for c in chunks]
            eval_res = rag_judge.evaluate_response(
                query=item["query"],
                answer=rag_out.get("answer", ""),
                contexts=contexts,
                use_llm_judge=req.use_llm_judge,
            )
            judge_rows.append({"id": item["id"], "query": item["query"], **eval_res})

        n_j = max(1, len(judge_rows))
        pass_count = sum(1 for r in judge_rows if r["verdict"] == "pass")
        hallucinated_count = sum(1 for r in judge_rows if r["hallucinated_references"])
        judge_summary = {
            "num_evaluated": len(judge_rows),
            "pass_rate": round(pass_count / n_j, 4),
            "mean_faithfulness": round(sum(r["faithfulness_score"] for r in judge_rows) / n_j, 4),
            "mean_answer_relevance": round(sum(r["answer_relevance_score"] for r in judge_rows) / n_j, 4),
            "mean_citation_grounding": round(sum(r["citation_grounding_score"] for r in judge_rows) / n_j, 4),
            "mean_overall_score": round(sum(r["overall_score"] for r in judge_rows) / n_j, 4),
            "hallucination_rate": round(hallucinated_count / n_j, 4),
            "per_query": judge_rows,
        }
        calibration = rag_judge.run_human_calibration_check()
        result["judge_summary"] = judge_summary
        result["calibration"] = calibration
        result["markdown_report"] = (
            markdown_report + "\n\n" + _format_judge_markdown(judge_summary, calibration)
        )

    return result


def _format_judge_markdown(judge_summary: Dict[str, Any], calibration: Dict[str, Any]) -> str:
    """Formats P2.1 LLM-as-a-Judge & Human Calibration summary into Markdown."""
    lines = [
        "### P2.1 LLM-as-a-Judge & Human-Calibrated Generation Audit",
        f"- **Human Calibration Agreement:** `{calibration['human_agreement_rate'] * 100:.1f}%` "
        f"(`{calibration['num_cases']}` golden cases | MAE: `{calibration['mean_absolute_error']:.3f}`)",
        f"- **Mean Faithfulness:** `{judge_summary['mean_faithfulness']:.3f}` | "
        f"**Answer Relevance:** `{judge_summary['mean_answer_relevance']:.3f}` | "
        f"**Citation Grounding:** `{judge_summary['mean_citation_grounding']:.3f}` | "
        f"**Pass Rate:** `{judge_summary['pass_rate'] * 100:.1f}%`",
        "",
        "| Query ID | Faithfulness | Relevance | Citation Grounding | Overall | Verdict | Hallucinated Refs |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for row in judge_summary["per_query"]:
        refs = ", ".join(row["hallucinated_references"]) if row["hallucinated_references"] else "None"
        lines.append(
            f"| `{row['id']}` | `{row['faithfulness_score']:.2f}` | `{row['answer_relevance_score']:.2f}` | "
            f"`{row['citation_grounding_score']:.2f}` | `{row['overall_score']:.2f}` | `{row['verdict'].upper()}` | {refs} |"
        )
    return "\n".join(lines)


def _format_markdown_ablation_table(summaries: List[Dict[str, Any]], k: int) -> str:
    """Formats a Staff-level Markdown comparison table for the Gradio Eval Studio."""
    lines = [
        f"### Retrieval Ablation Benchmark (`Top-K = {k}`, `N = {len(GOLDEN_QA_BENCHMARK)}` Golden Queries)",
        "",
        f"| Pipeline Mode | HitRate@{k} | MRR@{k} | NDCG@{k} | Context Precision@{k} | Mean Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    mode_labels = {
        "dense": "1. Dense Only (`all-MiniLM-L6-v2`)",
        "sparse": "2. Sparse Only (`BM25Okapi`)",
        "hybrid_rrf": "3. Hybrid RRF (`Dense + BM25, k=60`)",
        "hybrid_rerank": "4. Hybrid RRF + Cross-Encoder Reranker",
        "hybrid_graph_rerank": "5. **Tri-Modal GraphRAG (Dense + BM25 + Entity Graph + Reranker)**",
    }
    for row in summaries:
        label = mode_labels.get(row["mode"], row["mode"])
        lines.append(
            f"| {label} | `{row['hit_rate_at_k']:.3f}` | `{row['mrr_at_k']:.3f}` | "
            f"`{row['ndcg_at_k']:.3f}` | `{row['context_precision_at_k']:.3f}` | `{row['mean_latency_ms']:.2f} ms` |"
        )
    lines.append("")
    lines.append(
        "> **Staff RAG Takeaway:** Dense bi-encoders capture semantic paraphrases (`Q3`), whereas BM25 excels on exact error codes and rotation IDs (`Q1`, `Q2`). Combining both via **Reciprocal Rank Fusion (RRF)** maximizes candidate recall (`HitRate@K`), while the **Cross-Encoder Reranker** promotes the exact matching passage to Rank 1 (`MRR@K` & `NDCG@K`)."
    )
    return "\n".join(lines)
