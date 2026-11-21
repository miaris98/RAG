"""TDD tests for Branch 5: Perplexity Late Chunking embeddings (#9) & NVIDIA PDF layout/table parser (#10)."""

from fastapi.testclient import TestClient

from main import app
from app.ingestion.pdf_parser import extract_structured_markdown_from_pdf
from app.retrieval.hybrid import cosine_similarity, retrieval_engine


client = TestClient(app)


def _build_minimal_pdf(lines: list[str]) -> bytes:
    """Builds a minimal valid single-page PDF 1.4 containing the given text lines."""
    stream_ops = ["BT", "/F1 12 Tf", "50 750 Td", "16 TL"]
    for idx, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_ops.append(f"({escaped}) Tj" if idx == 0 else f"T* ({escaped}) Tj")
    stream_ops.append("ET")
    stream_bytes = "\n".join(stream_ops).encode("latin-1")

    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        (
            f"5 0 obj << /Length {len(stream_bytes)} >> stream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream endobj\n"
        ),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin-1"))
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )
    return bytes(out)


def test_perplexity_late_chunking_blends_document_boundary_context_into_dense_vectors():
    """#9 Perplexity: Late chunking blends full-document embedding into isolated chunk vectors."""
    retrieval_engine.clear()

    doc_content = (
        "# Project Hyperion Quantum Ledger\n\n"
        "Project Hyperion is our low-latency European clearing hub deployed in Zurich.\n\n"
        "## Cryptographic Rotation\n"
        "It enforces hardware key rotation every 14 days using isolated HSM enclaves."
    )

    # First ingest WITHOUT late chunking
    r_plain = client.post(
        "/ingest",
        json={
            "title": "Hyperion Plain",
            "source": "internal://hyperion-plain.md",
            "content": doc_content,
            "enable_late_chunking": False,
        },
    )
    assert r_plain.status_code == 200
    assert r_plain.json()["late_chunking_enabled"] is False
    plain_vecs = dict(retrieval_engine._dense_vectors)

    # Clear and ingest WITH late chunking enabled
    retrieval_engine.clear()
    r_late = client.post(
        "/ingest",
        json={
            "title": "Hyperion Late",
            "source": "internal://hyperion-late.md",
            "content": doc_content,
            "enable_late_chunking": True,
        },
    )
    assert r_late.status_code == 200
    assert r_late.json()["late_chunking_enabled"] is True

    late_chunks = list(retrieval_engine._chunks_by_id.values())
    assert len(late_chunks) >= 2
    assert all(c.metadata.get("late_chunking_applied") is True for c in late_chunks)

    # Verify late-chunked dense vectors shifted relative to plain vectors and dense retrieval finds the pronoun chunk
    rot_chunk = [c for c in late_chunks if "hardware key rotation" in c.content][0]
    assert rot_chunk.chunk_id in retrieval_engine._dense_vectors
    # Compare against plain vector for the matching snippet
    ret = client.post(
        "/retrieve",
        json={
            "query": "Project Hyperion Zurich hardware key rotation days",
            "retrieval_mode": "dense",
            "top_k": 3,
        },
    )
    assert ret.status_code == 200
    combined = " ".join(c["parent_content"] for c in ret.json()["results"])
    assert "14 days" in combined


def test_nvidia_pdf_parser_reconstructs_markdown_tables_and_ingests_via_endpoint():
    """#10 NVIDIA: PDF parser extracts layout pages + tabular rows into Markdown tables and ingests via POST /ingest/pdf."""
    retrieval_engine.clear()

    pdf_bytes = _build_minimal_pdf(
        [
            "Quarterly GPU Cluster SLO Report",
            "Cluster   Region   LatencyP95",
            "A100-EU   Frankfurt   42ms",
            "H100-US   Virginia   29ms",
        ]
    )
    parsed = extract_structured_markdown_from_pdf(pdf_bytes, filename="gpu_slo.pdf")
    assert parsed["page_count"] >= 1
    assert parsed["tables_detected"] >= 1
    assert "| Cluster | Region | LatencyP95 |" in parsed["markdown"]
    assert "| --- | --- | --- |" in parsed["markdown"]
    assert "H100-US" in parsed["markdown"]

    resp = client.post(
        "/ingest/pdf",
        files={"file": ("gpu_slo.pdf", pdf_bytes, "application/pdf")},
        data={
            "title": "GPU Cluster SLO PDF",
            "tenant_id": "default",
            "access_roles": "public,engineering",
            "enable_late_chunking": "true",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["pdf_page_count"] >= 1
    assert data["pdf_tables_detected"] >= 1
    assert data["late_chunking_enabled"] is True

    ret = client.post(
        "/retrieve",
        json={"query": "What is the LatencyP95 for H100-US in Virginia?", "retrieval_mode": "hybrid_rerank", "top_k": 3},
    )
    assert ret.status_code == 200
    assert ret.json()["count"] >= 1
    assert "29ms" in ret.json()["results"][0]["parent_content"]
