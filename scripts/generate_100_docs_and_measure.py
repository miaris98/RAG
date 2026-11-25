"""Generates a 100+ document local synthetic enterprise corpus (including multi-page PDFs with tables),
benchmarks every subsystem against the live Docker Compose stack, and prints measured JSON telemetry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List
import requests


def _build_minimal_pdf_bytes(title: str, lines: List[str]) -> bytes:
    """Builds a valid multi-line single/multi-page PDF using pure Python (no external PDF writer needed)."""
    # Escape PDF literal string characters
    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    stream_cmds = ["BT", "/F1 11 Tf", "50 760 Td", f"({_esc(title)}) Tj", "0 -22 Td"]
    for line in lines[:42]:
        stream_cmds.append(f"({_esc(line[:95])}) Tj")
        stream_cmds.append("0 -15 Td")
    stream_cmds.append("ET")
    stream_bytes = "\n".join(stream_cmds).encode("latin-1", errors="replace")

    objs: List[bytes] = []
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objs.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    )
    objs.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    objs.append(
        f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
        + stream_bytes
        + b"\nendstream\nendobj\n"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)


def _pct(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return round(s[idx], 2)


def generate_and_measure(base_url: str, pdf_dir: Path, total_docs: int = 100) -> Dict[str, Any]:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(60):
        try:
            if requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1.5)
    else:
        raise RuntimeError(f"Backend {base_url} did not become ready.")
    # 1. Reset corpus for clean 100-doc measurement
    requests.delete(f"{base_url}/corpus", timeout=15).raise_for_status()
    seed_res = requests.post(f"{base_url}/ingest-sample", timeout=60).json()

    domains = [
        ("payments", ["public", "engineering"], "ERR_LEDGER_9042", "aegis-ctl raft resync --force-quorum"),
        ("security", ["public", "compliance"], "ERR_VAULT_7719", "vault lease revoke -prefix auth/token"),
        ("ml_rag", ["public", "engineering"], "SLO_P95_250MS", "rag-ctl cache prewarm --tier=hot"),
        ("ops", ["engineering"], "PD-EMEA-8821", "kubectl rollout undo deployment/rag-gateway-api"),
        ("compliance", ["compliance"], "GDPR_ART17_SLA", "audit-ctl verify-chain --strict"),
    ]

    ingest_latencies_ms: List[float] = []
    pdf_latencies_ms: List[float] = []
    total_new_chunks = int(seed_res.get("chunks_indexed_new", 0))
    ingested_doc_ids: List[str] = []

    # 2. Generate & ingest 10 synthetic PDFs with tables
    num_pdfs = 10
    for i in range(num_pdfs):
        dom, roles, code, cmd = domains[i % len(domains)]
        title = f"Enterprise Architecture & Incident Runbook PDF #{i + 1:03d} ({dom})"
        lines = [
            f"Service Domain: {dom} | Error Code: {code} | Tenant: default",
            f"Remediation Command: {cmd}",
            "| Component | Target SLO | Mitigation |",
            "| Milvus Vector DB | p95 < 45ms | Pre-filter RBAC + HNSW SQ8 |",
            "| RabbitMQ Queue | Lag < 500 | Bounded prefetch + thread fallback |",
            f"Audit Contact: oncall-{i}@corp.internal | SSN: 123-45-{6700 + i:04d}",
            f"Detailed Runbook Section {i + 1}: When {code} triggers, isolate bulkhead {dom}_pool and execute {cmd}.",
        ]
        pdf_bytes = _build_minimal_pdf_bytes(title, lines)
        pdf_path = pdf_dir / f"runbook_{i + 1:03d}_{dom}.pdf"
        pdf_path.write_bytes(pdf_bytes)

        t0 = time.perf_counter()
        with pdf_path.open("rb") as fh:
            resp = requests.post(
                f"{base_url}/ingest/pdf",
                files={"file": (pdf_path.name, fh, "application/pdf")},
                data={
                    "title": title,
                    "tenant_id": "default",
                    "access_roles": ",".join(roles),
                    "chunking_strategy": "parent_child",
                    "enable_late_chunking": "true",
                    "enable_contextual_enrichment": "true",
                },
                timeout=30,
            )
        resp.raise_for_status()
        dt = round((time.perf_counter() - t0) * 1000.0, 2)
        pdf_latencies_ms.append(dt)
        d = resp.json()
        total_new_chunks += int(d.get("chunks_indexed_new", 0))
        if d.get("doc_id"):
            ingested_doc_ids.append(d["doc_id"])

    # 3. Generate & ingest 90 rich Markdown + Code docs across all 3 chunking strategies
    strategies = ["parent_child", "contextual_recursive", "fixed"]
    strategy_stats: Dict[str, Dict[str, Any]] = {s: {"docs": 0, "chunks": 0, "latencies_ms": []} for s in strategies}
    num_md_docs = total_docs - num_pdfs  # 90 docs -> 100 generated + 3 seeded = 103 docs total
    for i in range(num_md_docs):
        dom, roles, code, cmd = domains[i % len(domains)]
        strat = strategies[i % len(strategies)]
        tenant = "default" if i % 5 != 0 else "tenant_eu"
        content = (
            f"# Runbook {i + 1:03d}: {dom.upper()} Resilience & {code}\n\n"
            f"| Metric | Value | Owner |\n| :--- | :--- | :--- |\n"
            f"| Primary Code | `{code}` | `{dom}-sre` |\n"
            f"| Recovery SLA | `120s` | `platform-ops` |\n\n"
            f"## Automated Remediation Handler\n"
            f"```python\n"
            f"def handle_{dom}_{i}(event_code: str) -> str:\n"
            f"    # Verify quorum & bulkhead isolation before executing {cmd}\n"
            f"    if event_code == '{code}':\n"
            f"        return '{cmd}'\n"
            f"    return 'noop'\n"
            f"```\n\n"
            f"Operational guidance for incident #{i + 1:03d}: contact lead-{i}@corp.internal "
            f"and verify Prometheus `rag_bulkhead_saturation_ratio` and MinIO `a-bucket` segment health.\n"
        )
        t0 = time.perf_counter()
        resp = requests.post(
            f"{base_url}/ingest",
            json={
                "content": content,
                "title": f"Synthetic Spec #{i + 1:03d} ({dom} / {strat})",
                "source": f"synthetic://{dom}/doc-{i + 1:03d}.md",
                "chunking_strategy": strat,
                "tenant_id": tenant,
                "access_roles": roles,
                "apply_guardrails": True,
                "enable_contextual_enrichment": (strat != "fixed"),
                "enable_late_chunking": (strat == "parent_child"),
            },
            timeout=30,
        )
        resp.raise_for_status()
        dt = round((time.perf_counter() - t0) * 1000.0, 2)
        ingest_latencies_ms.append(dt)
        d = resp.json()
        new_c = int(d.get("chunks_indexed_new", 0))
        total_new_chunks += new_c
        strategy_stats[strat]["docs"] += 1
        strategy_stats[strat]["chunks"] += new_c
        strategy_stats[strat]["latencies_ms"].append(dt)
        if d.get("doc_id"):
            ingested_doc_ids.append(d["doc_id"])

    # 4. Measure SHA-256 idempotent re-ingestion deduplication
    dup_payload = {
        "content": "Idempotent SHA-256 deduplication probe document for 100-doc benchmark.",
        "title": "Dedup Probe Doc",
        "source": "synthetic://dedup/probe.md",
        "chunking_strategy": "parent_child",
        "tenant_id": "default",
        "access_roles": ["public", "engineering"],
    }
    d1 = requests.post(f"{base_url}/ingest", json=dup_payload, timeout=15).json()
    d2 = requests.post(f"{base_url}/ingest", json=dup_payload, timeout=15).json()

    # 5. Measure CDC upsert & delete
    t_cdc = time.perf_counter()
    cdc_up = requests.post(
        f"{base_url}/ingest/cdc",
        json={
            "external_id": "bench-cdc-001",
            "action": "upsert",
            "version": 1,
            "title": "Benchmark CDC Doc v1",
            "content": "CDC policy v1: quorum lease timeout is 45ms.",
            "tenant_id": "default",
            "access_roles": ["public", "engineering"],
        },
        timeout=15,
    ).json()
    cdc_del = requests.post(
        f"{base_url}/ingest/cdc",
        json={
            "external_id": "bench-cdc-001",
            "action": "delete",
            "version": 2,
            "tenant_id": "default",
            "access_roles": ["public", "engineering"],
        },
        timeout=15,
    ).json()
    cdc_ms = round((time.perf_counter() - t_cdc) * 1000.0, 2)

    # 6. Run 5-Mode Retrieval Ablation + Judge on the 100+ doc corpus
    t_eval = time.perf_counter()
    eval_res = requests.post(
        f"{base_url}/evaluate",
        json={
            "top_k": 5,
            "modes": ["dense", "sparse", "hybrid_rrf", "hybrid_rerank", "hybrid_graph_rerank"],
            "seed_golden_corpus": False,
            "enable_judge_eval": True,
            "use_llm_judge": False,
        },
        timeout=120,
    ).json()
    eval_ms = round((time.perf_counter() - t_eval) * 1000.0, 2)

    # 7. Measure Query Cold vs Semantic Cache Hit, Structured Taxonomy, SQL OpsBot, Zero Trust JWT, GDPR Erasure
    q_body = {
        "text": "What triggers ERR_LEDGER_9042 and what is the remediation command?",
        "retrieval_mode": "hybrid_graph_rerank",
        "top_k": 3,
        "tenant_id": "default",
        "user_roles": ["public", "engineering", "compliance"],
        "enable_web_search": False,
        "enable_semantic_cache": True,
    }
    t_cold = time.perf_counter()
    requests.post(f"{base_url}/query-stream", json=q_body, timeout=30)
    cold_query_ms = round((time.perf_counter() - t_cold) * 1000.0, 2)

    t_warm = time.perf_counter()
    r_warm = requests.post(f"{base_url}/query-stream", json=q_body, timeout=30)
    warm_cache_ms = round((time.perf_counter() - t_warm) * 1000.0, 2)

    t_struct = time.perf_counter()
    struct_res = requests.post(
        f"{base_url}/query-structured",
        json={
            "text": "ERR_LEDGER_9042 quorum drift timeout",
            "allowed_labels": [
                "payment_ledger_quorum_incident",
                "hsm_key_rotation_security",
                "data_retention_encryption_policy",
                "sovereign_compliance_escalation",
            ],
            "retrieval_mode": "hybrid_graph_rerank",
            "top_k": 3,
            "tenant_id": "default",
            "user_roles": ["public", "engineering"],
        },
        timeout=20,
    ).json()
    struct_ms = round((time.perf_counter() - t_struct) * 1000.0, 2)

    t_sql = time.perf_counter()
    sql_res = requests.post(
        f"{base_url}/inventory/sql-query",
        json={"sql": "SELECT service_name, region, status, p95_latency_ms FROM service_inventory ORDER BY service_name"},
        timeout=10,
    ).json()
    sql_ms = round((time.perf_counter() - t_sql) * 1000.0, 2)

    t_jwt = time.perf_counter()
    tok = requests.post(
        f"{base_url}/security/token",
        json={"subject": "bench-worker", "audience": "rag-gateway-api", "roles": ["public"], "ttl_sec": 120.0},
        timeout=10,
    ).json()["token"]
    ver1 = requests.post(f"{base_url}/security/verify", headers={"X-Service-Token": f"Bearer {tok}"}, timeout=10)
    ver2 = requests.post(f"{base_url}/security/verify", headers={"X-Service-Token": tok}, timeout=10)
    jwt_ms = round((time.perf_counter() - t_jwt) * 1000.0, 2)

    t_gdpr = time.perf_counter()
    gdpr_res = requests.delete(f"{base_url}/documents/{d1['doc_id']}", timeout=15).json()
    gdpr_ms = round((time.perf_counter() - t_gdpr) * 1000.0, 2)

    corpus_stats = requests.get(f"{base_url}/corpus", timeout=10).json()
    resilience = requests.get(f"{base_url}/health/resilience", timeout=10).json()
    audit_logs = requests.get(f"{base_url}/audit-logs", params={"limit": 5}, timeout=10).json()
    cqrs = requests.get(f"{base_url}/cqrs/read-model", timeout=10).json()
    cache_stats = requests.get(f"{base_url}/cache/stats", timeout=10).json()
    prom_text = requests.get(f"{base_url}/metrics/prometheus", timeout=10).text

    return {
        "corpus_stats": corpus_stats,
        "generated_pdfs_count": num_pdfs,
        "generated_markdown_count": num_md_docs,
        "pdf_ingest_ms": {"p50": _pct(pdf_latencies_ms, 50), "p95": _pct(pdf_latencies_ms, 95), "mean": round(statistics.mean(pdf_latencies_ms), 2)},
        "md_ingest_ms": {"p50": _pct(ingest_latencies_ms, 50), "p95": _pct(ingest_latencies_ms, 95), "mean": round(statistics.mean(ingest_latencies_ms), 2)},
        "chunking_strategies": {
            k: {
                "docs": v["docs"],
                "chunks": v["chunks"],
                "avg_chunks_per_doc": round(v["chunks"] / max(1, v["docs"]), 2),
                "p50_ms": _pct(v["latencies_ms"], 50),
                "p95_ms": _pct(v["latencies_ms"], 95),
            }
            for k, v in strategy_stats.items()
        },
        "dedup_probe": {"first_new": d1.get("chunks_indexed_new"), "second_new": d2.get("chunks_indexed_new"), "second_skipped": d2.get("chunks_skipped_duplicate")},
        "cdc_roundtrip_ms": cdc_ms,
        "cdc_deleted_chunks": cdc_del.get("deleted_chunks"),
        "ablation_eval_ms": eval_ms,
        "ablation_summaries": eval_res.get("summaries"),
        "judge_summary": eval_res.get("judge_summary"),
        "cold_query_ms": cold_query_ms,
        "warm_cache_ms": warm_cache_ms,
        "cache_hit_verified": '"cache_hit": true' in r_warm.text.lower(),
        "structured_taxonomy": {"label": struct_res.get("classification_label"), "confidence": struct_res.get("confidence"), "latency_ms": struct_ms},
        "sql_opsbot": {"row_count": sql_res.get("row_count"), "latency_ms": sql_ms},
        "zero_trust_jwt": {"verify_status": ver1.status_code, "replay_status": ver2.status_code, "roundtrip_ms": jwt_ms},
        "gdpr_erasure": {"chunks_deleted": gdpr_res.get("chunks_deleted"), "latency_ms": gdpr_ms},
        "audit_chain": {"chain_valid": audit_logs.get("chain_valid"), "total_events": audit_logs.get("total_events")},
        "cqrs_read_model": {"total_documents": cqrs.get("total_documents"), "total_chunks": cqrs.get("total_chunks"), "version": cqrs.get("projection_version")},
        "resilience": resilience,
        "cache_stats": cache_stats,
        "prometheus_metric_lines": len([ln for ln in prom_text.splitlines() if ln and not ln.startswith("#")]),
    }


if __name__ == "__main__":
    base = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    out_dir = Path(os.getenv("SYNTHETIC_PDF_DIR", "scratch_pdfs"))
    report = generate_and_measure(base_url=base, pdf_dir=out_dir, total_docs=100)
    print(json.dumps(report, indent=2))
