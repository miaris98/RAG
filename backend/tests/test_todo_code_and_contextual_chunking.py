"""TDD tests for P0.3 completion: fenced code preservation, AST symbol extraction, and Anthropic contextual enrichment."""

from fastapi.testclient import TestClient

from main import app
from app.ingestion.chunking import ChunkingEngine, extract_code_symbols, extract_sections
from app.models import DocumentInput
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_fenced_code_blocks_preserved_and_comments_not_treated_as_headers():
    """Fenced code blocks must stay intact across splits and Python '#' comments inside fences must not split sections."""
    doc_md = """# Quorum Sync Service

## Implementation
The reconciliation daemon repairs drifted Raft nodes safely:

```python
# Internal quorum resync helper (must NOT become a Markdown section header!)
class QuorumReconciler:
    def compute_quorum_lag(self, leader_term: int, follower_term: int) -> int:
        # Calculate absolute term delta
        return abs(leader_term - follower_term)

    async def resync_replica_shard(self, shard_id: str) -> bool:
        return bool(shard_id)
```

After reconciliation finishes, metrics are flushed to Prometheus."""

    sections = extract_sections(doc_md)
    headings = [h for h, _ in sections]
    assert not any("Internal quorum resync helper" in h for h in headings)
    assert not any("Calculate absolute term delta" in h for h in headings)

    engine = ChunkingEngine(
        parent_chunk_size=650,
        parent_chunk_overlap=50,
        child_chunk_size=180,
        child_chunk_overlap=20,
    )
    chunks = engine.chunk_document(
        content=doc_md,
        title="Quorum Sync Spec",
        source="internal://code/quorum_reconciler.md",
        strategy="parent_child",
    )

    code_chunks = [c for c in chunks if "class QuorumReconciler" in c.content]
    assert len(code_chunks) >= 1
    # Verify the entire fenced block remained intact inside the child chunk
    code_chunk = code_chunks[0]
    assert "def compute_quorum_lag" in code_chunk.content
    assert "async def resync_replica_shard" in code_chunk.content
    assert code_chunk.content.count("```") == 2

    # Verify Python AST extracted class, sync method, and async method symbols
    symbols = code_chunk.metadata.get("code_symbols", [])
    assert "QuorumReconciler" in symbols
    assert "compute_quorum_lag" in symbols
    assert "resync_replica_shard" in symbols
    assert "python" in code_chunk.metadata.get("code_languages", [])


def test_extract_code_symbols_supports_python_and_polyglot_snippets():
    """AST handles Python and regex handles TypeScript/Go/SQL code fences."""
    snippet = """```typescript
export class LedgerRouter {
  routePayment(txId: string) { return txId; }
}
```
```go
func VerifyQuorumSignature(payload []byte) bool { return true }
```"""
    info = extract_code_symbols(snippet)
    assert "LedgerRouter" in info["symbols"]
    assert "VerifyQuorumSignature" in info["symbols"]
    assert "typescript" in info["languages"]
    assert "go" in info["languages"]


def test_anthropic_contextual_enrichment_via_ingest_and_retrieve_api():
    """Contextual enrichment prepends document-level context so isolated pronoun chunks remain retrievable."""
    retrieval_engine.clear()

    doc_text = """# Project Helios Cold-Tier Storage

Project Helios is the European archival ledger engine running on encrypted NVMe clusters in Frankfurt.

## Failover SLA
When primary replication stalls for more than 12 seconds, it automatically shifts traffic to standby zone eu-central-2b without dropping in-flight commits."""

    resp = client.post(
        "/ingest",
        json={
            "title": "Project Helios Storage Spec",
            "source": "internal://storage/helios.md",
            "content": doc_text,
            "chunking_strategy": "parent_child",
            "tenant_id": "default",
            "access_roles": ["public", "engineering"],
            "enable_contextual_enrichment": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["contextual_enrichment_enabled"] is True

    ret = client.post(
        "/retrieve",
        json={
            "query": "Which standby zone does Project Helios shift traffic to when replication stalls?",
            "retrieval_mode": "hybrid_rerank",
            "top_k": 3,
        },
    )
    assert ret.status_code == 200
    results = ret.json()["results"]
    assert len(results) >= 1
    top_chunk = results[0]
    assert "[Context:" in top_chunk["content"]
    assert "eu-central-2b" in top_chunk["parent_content"]
