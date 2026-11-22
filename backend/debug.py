"""CLI Diagnostic & Ablation Inspector for the Staff RAG Engine.

Can be executed directly inside the backend container or locally:
    python debug.py
"""

from app.evaluation.benchmark import run_ablation_benchmark
from app.ingestion.pipeline import ingestion_pipeline
from app.models import EvaluationRequest
from app.retrieval.embeddings import ModelRegistry
from app.retrieval.hybrid import retrieval_engine


def debug_search() -> None:
    print("=" * 72)
    print("STAFF RAG ENGINE: DIAGNOSTIC & TWO-STAGE HYBRID RETRIEVAL INSPECTOR")
    print("=" * 72)

    print("\n1. Initializing Singleton Model Registry...")
    embedder = ModelRegistry.get_embeddings()
    test_vec = embedder.embed_query("diagnostic vector dimension check")
    print(f"   -> Model Status: {ModelRegistry.status()}")
    print(f"   -> Embedding Vector Dimension: {len(test_vec)}")

    print("\n2. Seeding Enterprise Sample Corpus (Parent-Child Chunking + SHA-256 Dedup)...")
    seed_res = ingestion_pipeline.ingest_sample_corpus(strategy="parent_child")
    print(f"   -> New Chunks Indexed: {seed_res['chunks_indexed_new']}")
    print(f"   -> Duplicate Chunks Skipped: {seed_res['chunks_skipped_duplicate']}")
    print(f"   -> Corpus Stats: {retrieval_engine.stats()}")

    query = "What triggers ERR_LEDGER_9042 and what is the remediation command?"
    print(f"\n3. Inspecting Two-Stage Hybrid Retrieval + Cross-Encoder Reranking for:\n   '{query}'")
    results = retrieval_engine.retrieve(queries=[query], mode="hybrid_rerank", top_k=3)
    for idx, chunk in enumerate(results, start=1):
        print(
            f"   [{idx}] {chunk.title} ({chunk.chunk_id})\n"
            f"       DenseRank={chunk.dense_rank} (sim={chunk.dense_score}) | "
            f"BM25Rank={chunk.sparse_rank} (bm25={chunk.sparse_score}) | "
            f"RRF={chunk.rrf_score:.5f} | CrossEncoderProb={chunk.calibrated_prob:.4f} | "
            f"Relevant={chunk.is_relevant}"
        )

    print("\n4. Running Golden QA Retrieval Ablation Benchmark...")
    eval_report = run_ablation_benchmark(EvaluationRequest(top_k=3))
    print("\n" + eval_report["markdown_report"])


if __name__ == "__main__":
    debug_search()
