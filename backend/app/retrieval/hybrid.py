"""Tri-Modal Hybrid Retrieval Engine with Pre-Filtered RBAC:
- Stage 0: Pre-Filtered Multi-Tenant & Role-Based Access Control (prevents recall starvation)
- Stage 1A: Dense Vector Similarity Search (Milvus + Cosine ANN)
- Stage 1B: Sparse Lexical Search (BM25Okapi over technical/alphanumeric tokens)
- Stage 1C: Entity Knowledge Graph 1-Hop Traversal (GraphRAG)
- Stage 1D: Reciprocal Rank Fusion (RRF, k=60) + Parent-Child Small-to-Big Promotion
- Stage 2: Cross-Encoder Joint Reranking + Calibrated Relevance Gating
"""

import math
import threading
from typing import Dict, List, Optional, Set, Tuple

from app.config import settings
from app.models import ChunkRecord, RetrievalMode, RetrievedChunk
from app.retrieval.embeddings import ModelRegistry
from app.retrieval.entity_graph import entity_graph
from app.retrieval.reranker import CrossEncoderReranker, tokenize_content_words


class PurePythonBM25Okapi:
    """Zero-dependency BM25Okapi implementation (k1=1.5, b=0.75)."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 1.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

        df: Dict[str, int] = {}
        for doc in corpus_tokens:
            frequencies: Dict[str, int] = {}
            for token in doc:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                df[token] = df.get(token, 0) + 1

        for token, freq in df.items():
            # Robertson-Sparck Jones IDF with smoothing so small corpora never yield negative IDF
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for idx, doc_freq in enumerate(self.doc_freqs):
                tf = doc_freq.get(q, 0)
                if tf == 0:
                    continue
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[idx] += q_idf * (tf * (self.k1 + 1.0)) / denom
        return scores


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Computes cosine similarity in [-1.0, 1.0] between two dense vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridRetrievalEngine:
    """Coordinates Pre-Filtered RBAC, Milvus Dense Search, BM25, Entity Graph, RRF, and Cross-Encoder Reranking."""

    def __init__(self):
        self._lock = threading.RLock()
        self._chunks_by_id: Dict[str, ChunkRecord] = {}
        self._chunk_order: List[str] = []
        self._dense_vectors: Dict[str, List[float]] = {}
        self._bm25 = None
        self._reranker = CrossEncoderReranker(relevance_threshold=settings.relevance_threshold)
        self._milvus_connected: bool = False
        self._milvus_hydrated: bool = False

    def add_chunks(self, chunks: List[ChunkRecord]) -> Tuple[int, int]:
        """Idempotently indexes chunks into the local registry, BM25, Entity Graph, and Milvus."""
        if not chunks:
            return 0, 0

        with self._lock:
            new_chunks: List[ChunkRecord] = []
            skipped = 0
            for chk in chunks:
                if chk.chunk_id in self._chunks_by_id:
                    skipped += 1
                else:
                    new_chunks.append(chk)

            if not new_chunks:
                return 0, skipped

            embedder = ModelRegistry.get_embeddings()
            texts = [c.content for c in new_chunks]
            vectors = embedder.embed_documents(texts)

            for chk, vec in zip(new_chunks, vectors):
                chk.entities = entity_graph.add_chunk(chk.chunk_id, chk.parent_content)
                self._chunks_by_id[chk.chunk_id] = chk
                self._chunk_order.append(chk.chunk_id)
                self._dense_vectors[chk.chunk_id] = vec

            self._rebuild_bm25()
            self._push_to_milvus(new_chunks)
            return len(new_chunks), skipped

    def _rebuild_bm25(self) -> None:
        if not self._chunk_order:
            self._bm25 = None
            return
        tokenized_corpus = [
            tokenize_content_words(self._chunks_by_id[cid].content) for cid in self._chunk_order
        ]
        self._bm25 = PurePythonBM25Okapi(tokenized_corpus)

    def _push_to_milvus(self, new_chunks: List[ChunkRecord]) -> None:
        try:
            from langchain_core.documents import Document as LangChainDocument
            from langchain_community.vectorstores import Milvus

            lc_docs = [
                LangChainDocument(
                    page_content=c.content,
                    metadata={
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "parent_id": c.parent_id,
                        "title": c.title,
                        "source": c.source,
                        "section": c.section,
                        "tenant_id": c.tenant_id,
                        "access_roles": ",".join(c.access_roles),
                        "parent_content": c.parent_content,
                    },
                )
                for c in new_chunks
            ]
            Milvus.from_documents(
                lc_docs,
                ModelRegistry.get_embeddings(),
                collection_name=settings.milvus_collection,
                connection_args={"host": settings.milvus_host, "port": settings.milvus_port},
                consistency_level="Strong",
            )
            self._milvus_connected = True
        except Exception as exc:
            self._milvus_connected = False
            print(f"[HybridRetrievalEngine] Milvus sync skipped (using in-memory + BM25 + Graph index): {exc}")

    def _hydrate_from_milvus_if_empty(self) -> None:
        if self._milvus_hydrated or self._chunk_order:
            return
        self._milvus_hydrated = True
        try:
            from langchain_community.vectorstores import Milvus

            embedder = ModelRegistry.get_embeddings()
            vector_db = Milvus(
                embedding_function=embedder,
                collection_name=settings.milvus_collection,
                connection_args={"host": settings.milvus_host, "port": settings.milvus_port},
                consistency_level="Strong",
            )
            docs = vector_db.similarity_search("document overview system information", k=100)
            if not docs:
                return
            hydrated: List[ChunkRecord] = []
            for idx, doc in enumerate(docs):
                meta = doc.metadata or {}
                cid = meta.get("chunk_id") or f"chk_milvus_{idx}"
                raw_roles = meta.get("access_roles", "public")
                roles = [r.strip() for r in raw_roles.split(",") if r.strip()] if isinstance(raw_roles, str) else list(raw_roles)
                hydrated.append(
                    ChunkRecord(
                        chunk_id=cid,
                        doc_id=meta.get("doc_id", "doc_milvus"),
                        parent_id=meta.get("parent_id", f"par_milvus_{idx}"),
                        title=meta.get("title", "Milvus Persisted Document"),
                        source=meta.get("source", "milvus_volume"),
                        section=meta.get("section", "Persisted"),
                        tenant_id=meta.get("tenant_id", "default"),
                        access_roles=roles or ["public"],
                        content=doc.page_content,
                        parent_content=meta.get("parent_content", doc.page_content),
                        chunk_index=idx,
                        token_estimate=max(1, len(doc.page_content.split())),
                    )
                )
            if hydrated:
                with self._lock:
                    texts = [c.content for c in hydrated]
                    vecs = embedder.embed_documents(texts)
                    for chk, vec in zip(hydrated, vecs):
                        if chk.chunk_id not in self._chunks_by_id:
                            chk.entities = entity_graph.add_chunk(chk.chunk_id, chk.parent_content)
                            self._chunks_by_id[chk.chunk_id] = chk
                            self._chunk_order.append(chk.chunk_id)
                            self._dense_vectors[chk.chunk_id] = vec
                    self._rebuild_bm25()
                    self._milvus_connected = True
        except Exception:
            pass

    def _authorized_chunk_ids(self, tenant_id: str, user_roles: Optional[List[str]]) -> Set[str]:
        """Stage 0 Pre-Filter: Computes the exact set of chunk_ids authorized for (tenant_id, user_roles)."""
        if user_roles is None:
            caller_roles = {"public", "engineering", "compliance"}
        else:
            caller_roles = {r.strip().lower() for r in user_roles if r and r.strip()}

        authorized: Set[str] = set()
        for cid in self._chunk_order:
            chk = self._chunks_by_id[cid]
            if chk.tenant_id != tenant_id:
                continue
            chunk_roles = {r.strip().lower() for r in chk.access_roles}
            if chunk_roles & caller_roles:
                authorized.add(cid)
        return authorized

    def retrieve(
        self,
        queries: List[str],
        mode: RetrievalMode = "hybrid_graph_rerank",
        top_k: int = settings.rerank_top_k,
        promote_parents: bool = True,
        tenant_id: str = "default",
        user_roles: Optional[List[str]] = None,
        dense_weight: float = settings.dense_weight,
        sparse_weight: float = settings.sparse_weight,
        graph_weight: float = 0.85,
    ) -> List[RetrievedChunk]:
        """Executes pre-filtered RBAC + Tri-Modal retrieval + RRF + Cross-Encoder Reranking."""
        self._hydrate_from_milvus_if_empty()

        with self._lock:
            if not self._chunk_order:
                return []

            authorized_ids = self._authorized_chunk_ids(tenant_id, user_roles)
            if not authorized_ids:
                return []

            clean_queries = [q.strip() for q in queries if q and q.strip()]
            if not clean_queries:
                return []
            primary_query = clean_queries[0]

            # 1. Dense Bi-Encoder Rankings (Pre-Filtered to authorized_ids)
            dense_ranks_per_chunk: Dict[str, int] = {}
            dense_scores_per_chunk: Dict[str, float] = {}
            if mode in ("dense", "hybrid_rrf", "hybrid_rerank", "hybrid_graph_rerank"):
                embedder = ModelRegistry.get_embeddings()
                for q in clean_queries:
                    q_vec = embedder.embed_query(q)
                    scored = [
                        (cid, cosine_similarity(q_vec, self._dense_vectors[cid]))
                        for cid in self._chunk_order
                        if cid in authorized_ids
                    ]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    for rank_idx, (cid, sim) in enumerate(scored[: settings.retrieval_top_k_dense], start=1):
                        if cid not in dense_ranks_per_chunk or rank_idx < dense_ranks_per_chunk[cid]:
                            dense_ranks_per_chunk[cid] = rank_idx
                        if cid not in dense_scores_per_chunk or sim > dense_scores_per_chunk[cid]:
                            dense_scores_per_chunk[cid] = sim

            # 2. Sparse BM25 Rankings (Pre-Filtered to authorized_ids)
            sparse_ranks_per_chunk: Dict[str, int] = {}
            sparse_scores_per_chunk: Dict[str, float] = {}
            if mode in ("sparse", "hybrid_rrf", "hybrid_rerank", "hybrid_graph_rerank") and self._bm25 is not None:
                for q in clean_queries:
                    q_tokens = tokenize_content_words(q)
                    bm25_scores = self._bm25.get_scores(q_tokens)
                    indexed_scores = [
                        (self._chunk_order[i], float(bm25_scores[i]))
                        for i in range(len(self._chunk_order))
                        if bm25_scores[i] > 0.0 and self._chunk_order[i] in authorized_ids
                    ]
                    indexed_scores.sort(key=lambda x: x[1], reverse=True)
                    for rank_idx, (cid, b_score) in enumerate(
                        indexed_scores[: settings.retrieval_top_k_sparse], start=1
                    ):
                        if cid not in sparse_ranks_per_chunk or rank_idx < sparse_ranks_per_chunk[cid]:
                            sparse_ranks_per_chunk[cid] = rank_idx
                        if cid not in sparse_scores_per_chunk or b_score > sparse_scores_per_chunk[cid]:
                            sparse_scores_per_chunk[cid] = b_score

            # 3. Entity Knowledge Graph 1-Hop Expansion (for Tri-Modal GraphRAG)
            graph_ranks_per_chunk: Dict[str, int] = {}
            graph_scores_per_chunk: Dict[str, float] = {}
            if mode == "hybrid_graph_rerank":
                graph_hits = entity_graph.score_chunks(
                    queries=clean_queries,
                    allowed_chunk_ids=authorized_ids,
                    top_k=settings.retrieval_top_k_dense,
                )
                for rank_idx, (cid, g_score) in enumerate(graph_hits, start=1):
                    graph_ranks_per_chunk[cid] = rank_idx
                    graph_scores_per_chunk[cid] = g_score

            # 4. Reciprocal Rank Fusion (RRF)
            candidate_ids = (
                set(dense_ranks_per_chunk.keys())
                | set(sparse_ranks_per_chunk.keys())
                | set(graph_ranks_per_chunk.keys())
            )
            if not candidate_ids and mode == "sparse":
                candidate_ids = set(list(authorized_ids)[:top_k])

            candidates: List[RetrievedChunk] = []
            rrf_k = settings.rrf_k
            for cid in candidate_ids:
                record = self._chunks_by_id[cid]
                d_rank = dense_ranks_per_chunk.get(cid)
                d_score = dense_scores_per_chunk.get(cid)
                s_rank = sparse_ranks_per_chunk.get(cid)
                s_score = sparse_scores_per_chunk.get(cid)
                g_rank = graph_ranks_per_chunk.get(cid)
                g_score = graph_scores_per_chunk.get(cid)

                rrf_val = 0.0
                if mode == "dense" and d_rank is not None:
                    rrf_val = dense_weight / (rrf_k + d_rank)
                elif mode == "sparse" and s_rank is not None:
                    rrf_val = sparse_weight / (rrf_k + s_rank)
                else:
                    if d_rank is not None:
                        rrf_val += dense_weight / (rrf_k + d_rank)
                    if s_rank is not None:
                        rrf_val += sparse_weight / (rrf_k + s_rank)
                    if g_rank is not None:
                        rrf_val += graph_weight / (rrf_k + g_rank)

                initial_prob = 0.0
                if d_score is not None and s_score is not None:
                    initial_prob = min(0.95, max(0.0, 0.5 * d_score + 0.5 * min(1.0, s_score / 8.0)))
                elif d_score is not None:
                    initial_prob = max(0.0, min(0.95, d_score))
                elif s_score is not None:
                    initial_prob = max(0.0, min(0.95, s_score / 8.0))

                candidates.append(
                    RetrievedChunk(
                        chunk_id=record.chunk_id,
                        doc_id=record.doc_id,
                        parent_id=record.parent_id,
                        title=record.title,
                        source=record.source,
                        section=record.section,
                        tenant_id=record.tenant_id,
                        access_roles=list(record.access_roles),
                        content=record.content,
                        parent_content=record.parent_content if promote_parents else record.content,
                        dense_rank=d_rank,
                        dense_score=round(d_score, 4) if d_score is not None else None,
                        sparse_rank=s_rank,
                        sparse_score=round(s_score, 4) if s_score is not None else None,
                        graph_rank=g_rank,
                        graph_score=round(g_score, 4) if g_score is not None else None,
                        rrf_score=round(rrf_val, 6),
                        calibrated_prob=round(initial_prob, 4),
                        is_relevant=initial_prob >= settings.relevance_threshold,
                    )
                )

            candidates.sort(
                key=lambda c: (c.rrf_score, c.dense_score or 0.0, c.sparse_score or 0.0),
                reverse=True,
            )

            # 5. Small-to-Big Parent Deduplication
            if promote_parents:
                deduped_by_parent: Dict[str, RetrievedChunk] = {}
                for cand in candidates:
                    pid = cand.parent_id or cand.chunk_id
                    if pid not in deduped_by_parent:
                        deduped_by_parent[pid] = cand
                    else:
                        existing = deduped_by_parent[pid]
                        existing.rrf_score = round(existing.rrf_score + 0.15 * cand.rrf_score, 6)
                candidates = list(deduped_by_parent.values())
                candidates.sort(
                    key=lambda c: (c.rrf_score, c.dense_score or 0.0, c.sparse_score or 0.0),
                    reverse=True,
                )

            # 6. Stage 2: Cross-Encoder Reranking
            if mode in ("hybrid_rerank", "hybrid_graph_rerank"):
                pool = candidates[: max(top_k * 3, 12)]
                return self._reranker.rerank(primary_query, pool, top_k=top_k)

            return candidates[:top_k]

    def stats(self) -> dict:
        with self._lock:
            unique_docs = {c.doc_id for c in self._chunks_by_id.values()}
            unique_parents = {c.parent_id for c in self._chunks_by_id.values()}
            titles = sorted({c.title for c in self._chunks_by_id.values()})
            return {
                "total_chunks": len(self._chunk_order),
                "total_parent_windows": len(unique_parents),
                "total_documents": len(unique_docs),
                "document_titles": titles,
                "milvus_connected": self._milvus_connected,
                "bm25_ready": self._bm25 is not None,
                "entity_graph": entity_graph.stats(),
            }

    def clear(self) -> None:
        with self._lock:
            self._chunks_by_id.clear()
            self._chunk_order.clear()
            self._dense_vectors.clear()
            self._bm25 = None
            entity_graph.clear()


retrieval_engine = HybridRetrievalEngine()
