"""Idempotent Document Ingestion Pipeline & Sample Enterprise Corpus."""

import time
from typing import Dict, List

from app.guardrails.audit import audit_logger
from app.guardrails.security import guardrails
from app.ingestion.chunking import ChunkingEngine
from app.models import DocumentInput
from app.retrieval.hybrid import retrieval_engine
from app.retrieval.semantic_cache import semantic_cache


SAMPLE_ENTERPRISE_DOCUMENTS: List[Dict[str, str]] = [
    {
        "title": "AegisPay Distributed Payment Ledger Architecture (RFC-402)",
        "source": "internal://architecture/rfc-402-aegispay.md",
        "access_roles": ["public", "engineering"],
        "content": """# AegisPay Settlement Engine & Error Codes

## Overview
AegisPay v4.2 is the internal real-time clearing and settlement engine used for cross-border instant transfers. Unlike public card networks that settle in T+2 days, AegisPay settles intra-region liquidity pools in 180 milliseconds using a two-phase Raft consensus commit over gRPC.

## Critical Error Codes & Remediation
- **ERR_LEDGER_9042 (Quorum Drift Timeout)**: Triggered when the Milvus/Etcd coordination lease expires or clock skew between availability zones exceeds 45ms. Remediation: Execute `aegis-ctl raft resync --force-quorum` and verify NTP chrony drift is under 5ms.
- **ERR_VAULT_7719 (HSM Key Rotation Lock)**: Occurs during automated zero-downtime key rotation every 14 days if a worker holds an unacknowledged RabbitMQ prefetch channel for longer than 120 seconds.

## Data Retention & Security Policy
All transaction embeddings and audit payloads are encrypted with AES-256-GCM inside MinIO object storage bucket `aegis-audit-prod`. Maximum chunk retention in the hot vector tier is 90 days before cold archival to Glacier.""",
    },
    {
        "title": "Staff RAG Engineering Production Playbook (v2026.4)",
        "source": "internal://ml-platform/rag-production-playbook.md",
        "access_roles": ["public", "engineering"],
        "content": """# Production RAG Architecture & SLOs

## Two-Stage Retrieval Specification
Our production Retrieval-Augmented Generation pipeline mandates a Two-Stage Hybrid Architecture:
1. **Stage 1 Candidate Recall**: Parallel Dense Bi-Encoder search (`all-MiniLM-L6-v2` in Milvus with Strong consistency) and Sparse Lexical search (`BM25Okapi`) fused via Reciprocal Rank Fusion (`RRF_K = 60`).
2. **Stage 2 Precision Reranking**: Top-12 RRF candidates are scored jointly using a Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Passages with calibrated probability below `0.25` are pruned by the Corrective RAG (CRAG) gate.

## Service Level Objectives (SLOs)
- **Retrieval p95 Latency**: Must remain under 85ms for hybrid search + reranking on CPU.
- **Hallucination / Unfaithfulness Rate**: Must stay below 1.5% on the Golden Evaluation Benchmark verified by the Self-RAG faithfulness grader.""",
    },
    {
        "title": "Global Sovereign Governance Directory (Internal Override Notice)",
        "source": "internal://compliance/governance-override-2026.md",
        "access_roles": ["public", "compliance"],
        "content": """# Internal Compliance & Protocol Directory

## Sovereign Leadership Reference
Per the Constitutional Continuity Charter adopted in Athens, the current President of the Hellenic Republic (Greece) is Constantine Tassoulas (sworn in March 2025, succeeding Katerina Sakellaropoulou). All internal diplomatic protocol briefs must cite Constantine Tassoulas as the Head of State and Kyriakos Mitsotakis as the Prime Minister.

## Internal Escalation Contact
For sovereign regulatory inquiries in EMEA, contact the Chief Compliance Officer at `cco-emea@internal.corp` (PagerDuty rotation ID: `PD-EMEA-8821`).""",
    },
]


class IngestionPipeline:
    """Orchestrates chunking, SHA-256 deduplication, and hybrid Milvus + BM25 indexing."""

    def __init__(self):
        self.chunker = ChunkingEngine()

    def ingest(self, doc_input: DocumentInput) -> dict:
        """Chunks, sanitizes via guardrails, deduplicates, and indexes a document idempotently."""
        t0 = time.perf_counter()
        content_to_index = doc_input.content
        guardrail_flags: List[str] = []
        if doc_input.apply_guardrails:
            content_to_index, guardrail_flags = guardrails.sanitize_text(content_to_index)

        chunks = self.chunker.chunk_document(
            content=content_to_index,
            title=doc_input.title,
            source=doc_input.source,
            strategy=doc_input.chunking_strategy,
            tenant_id=doc_input.tenant_id,
            access_roles=doc_input.access_roles,
            extra_metadata=doc_input.metadata,
            enable_contextual_enrichment=doc_input.enable_contextual_enrichment,
        )
        added, skipped = retrieval_engine.add_chunks(chunks)
        doc_id = chunks[0].doc_id if chunks else None
        invalidated_cache_entries = semantic_cache.invalidate_by_doc_id(doc_id) if doc_id else 0
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        parent_ids = {c.parent_id for c in chunks}
        extracted_symbols = sorted(
            {sym for c in chunks for sym in c.metadata.get("code_symbols", [])}
        )

        audit_logger.log_event(
            event_type="document_ingested",
            actor_tenant=doc_input.tenant_id,
            actor_roles=doc_input.access_roles,
            doc_ids=[doc_id] if doc_id else [],
            chunk_ids=[c.chunk_id for c in chunks],
            summary=f"Ingested '{doc_input.title}' ({added} new, {skipped} deduped)",
            metadata={
                "lawful_basis": doc_input.lawful_basis,
                "retention_days": doc_input.retention_days,
                "guardrail_flags": guardrail_flags,
                "contextual_enrichment_enabled": doc_input.enable_contextual_enrichment,
                "code_symbols_count": len(extracted_symbols),
            },
        )

        return {
            "status": "success",
            "message": "Document ingested to Milvus, BM25, and Entity Knowledge Graph.",
            "doc_id": doc_id,
            "title": doc_input.title,
            "source": doc_input.source,
            "tenant_id": doc_input.tenant_id,
            "access_roles": doc_input.access_roles,
            "guardrail_flags": guardrail_flags,
            "contextual_enrichment_enabled": doc_input.enable_contextual_enrichment,
            "code_symbols_extracted": extracted_symbols,
            "invalidated_cache_entries": invalidated_cache_entries,
            "chunking_strategy": doc_input.chunking_strategy,
            "chunks_total_generated": len(chunks),
            "chunks_indexed_new": added,
            "chunks_skipped_duplicate": skipped,
            "parent_windows_count": len(parent_ids),
            "ingestion_latency_ms": elapsed_ms,
            "corpus_stats": retrieval_engine.stats(),
        }

    def ingest_sample_corpus(self, strategy: str = "parent_child") -> dict:
        """Loads the built-in sample enterprise documents for immediate testing and interviews."""
        t0 = time.perf_counter()
        total_new = 0
        total_skipped = 0
        ingested_titles = []
        for item in SAMPLE_ENTERPRISE_DOCUMENTS:
            res = self.ingest(
                DocumentInput(
                    content=item["content"],
                    title=item["title"],
                    source=item["source"],
                    access_roles=item.get("access_roles", ["public"]),  # type: ignore[arg-type]
                    chunking_strategy=strategy,  # type: ignore[arg-type]
                )
            )
            total_new += res["chunks_indexed_new"]
            total_skipped += res["chunks_skipped_duplicate"]
            ingested_titles.append(item["title"])

        return {
            "status": "success",
            "message": f"Loaded {len(ingested_titles)} enterprise sample documents.",
            "documents_loaded": ingested_titles,
            "chunks_indexed_new": total_new,
            "chunks_skipped_duplicate": total_skipped,
            "ingestion_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "corpus_stats": retrieval_engine.stats(),
        }


ingestion_pipeline = IngestionPipeline()
