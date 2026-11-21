"""TDD tests for Branch 6: Spotify hierarchical Semantic IDs (#14) & Airbnb tenant/partition shard routing (#20)."""

from fastapi.testclient import TestClient

from main import app
from app.ingestion.pipeline import ingestion_pipeline
from app.retrieval.hybrid import retrieval_engine


client = TestClient(app)


def test_spotify_hierarchical_semantic_ids_assigned_and_filterable():
    """#14 Spotify: every chunk gets a hierarchical Semantic ID (sem.<domain>.<topic>.cXX) filterable on /retrieve."""
    retrieval_engine.clear()
    ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")

    chunks = list(retrieval_engine._chunks_by_id.values())
    assert len(chunks) >= 6
    for chk in chunks:
        assert chk.semantic_id.startswith("sem.")
        parts = chk.semantic_id.split(".")
        assert len(parts) == 4, f"Expected 4-part hierarchical semantic ID, got {chk.semantic_id}"
        assert parts[3].startswith("c")

    # Verify prefix filtering narrows retrieval cleanly to matching semantic cluster
    sample_prefix = ".".join(chunks[0].semantic_id.split(".")[:2])
    ret = client.post(
        "/retrieve",
        json={
            "query": chunks[0].title,
            "retrieval_mode": "hybrid_rerank",
            "top_k": 5,
            "semantic_id_prefix": sample_prefix,
        },
    )
    assert ret.status_code == 200
    results = ret.json()["results"]
    assert len(results) >= 1
    assert all(r["semantic_id"].startswith(sample_prefix) for r in results)


def test_airbnb_tenant_partition_shard_routing_prunes_foreign_shards():
    """#20 Airbnb: sharded scatter-gather routes strictly to caller tenant shards and prunes foreign shards."""
    retrieval_engine.clear()

    client.post(
        "/ingest",
        json={
            "title": "Tenant Alpha Runbook",
            "content": "Tenant Alpha uses isolated shard routing for sub-20ms scatter-gather vector search.",
            "tenant_id": "tenant_alpha",
            "access_roles": ["public"],
        },
    )
    client.post(
        "/ingest",
        json={
            "title": "Tenant Beta Runbook",
            "content": "Tenant Beta stores separate payment partition keys for EMEA settlement.",
            "tenant_id": "tenant_beta",
            "access_roles": ["public"],
        },
    )

    ret = client.post(
        "/retrieve",
        json={
            "query": "isolated shard routing scatter-gather vector search",
            "retrieval_mode": "hybrid_rerank",
            "top_k": 3,
            "tenant_id": "tenant_alpha",
            "user_roles": ["public"],
        },
    )
    assert ret.status_code == 200
    data = ret.json()
    funnel = data["retrieval_funnel"]
    assert len(funnel["routed_shards"]) >= 1
    assert all(s.startswith("shard_tenant_alpha_") for s in funnel["routed_shards"])
    assert funnel["pruned_shards_count"] >= 1
    assert all(r["tenant_id"] == "tenant_alpha" for r in data["results"])

    # Verify observability endpoint exposes shard & semantic cluster topology
    topo = client.get("/corpus/shards-and-clusters")
    assert topo.status_code == 200
    t_data = topo.json()
    assert t_data["total_shards"] >= 2
    assert t_data["total_semantic_clusters"] >= 1
