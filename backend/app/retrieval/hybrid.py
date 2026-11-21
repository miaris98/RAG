"""Tri-Modal Hybrid Retrieval Engine with Pre-Filtered RBAC:
- Stage 0: Pre-Filtered Multi-Tenant & Role-Based Access Control (prevents recall starvation)
- Stage 1A: Dense Vector Similarity Search (Milvus + Cosine ANN)
- Stage 1B: Sparse Lexical Search (BM25Okapi over technical/alphanumeric tokens)
- Stage 1C: Entity Knowledge Graph 1-Hop Traversal (GraphRAG)
- Stage 1D: Reciprocal Rank Fusion (RRF, k=60) + Parent-Child Small-to-Big Promotion
- Stage 2: Cross-Encoder Joint Reranking + Calibrated Relevance Gating
"""

import math
import socket
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.models import ChunkRecord, RetrievalMode, RetrievedChunk
from app.retrieval.embeddings import ModelRegistry
from app.retrieval.entity_graph import entity_graph, extract_entities
from app.retrieval.postprocessor import context_postprocessor
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
        self._last_funnel: Dict[str, Any] = {}

    def get_last_funnel(self) -> Dict[str, Any]:
        """Returns the retrieval candidate funnel and substage timing telemetry from the most recent retrieve()."""
        with self._lock:
            return dict(self._last_funnel)

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

    @staticmethod
    def _is_milvus_reachable(timeout_sec: float = 0.25) -> bool:
        try:
            with socket.create_connection((settings.milvus_host, int(settings.milvus_port)), timeout=timeout_sec):
                return True
        except OSError:
            return False

    def _push_to_milvus(self, new_chunks: List[ChunkRecord]) -> None:
        if not self._is_milvus_reachable():
            self._milvus_connected = False
            return
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
            if not client.has_collection(settings.milvus_collection):
                client.create_collection(
                    collection_name=settings.milvus_collection,
                    dimension=len(self._dense_vectors[new_chunks[0].chunk_id]),
                    metric_type="COSINE",
                    auto_id=False,
                )
            rows = [
                {
                    "id": c.chunk_id,
                    "vector": self._dense_vectors[c.chunk_id],
                    "doc_id": c.doc_id,
                    "parent_id": c.parent_id,
                    "title": c.title,
                    "source": c.source,
                    "section": c.section,
                    "tenant_id": c.tenant_id,
                    "access_roles": ",".join(c.access_roles),
                    "content": c.content,
                    "parent_content": c.parent_content,
                }
                for c in new_chunks
            ]
            client.upsert(collection_name=settings.milvus_collection, data=rows)
            self._milvus_connected = True
        except Exception as exc:
            self._milvus_connected = False
            print(f"[HybridRetrievalEngine] Milvus sync skipped (using in-memory + BM25 + Graph index): {exc}")

    def _hydrate_from_milvus_if_empty(self) -> None:
        if self._milvus_hydrated or self._chunk_order:
            return
        self._milvus_hydrated = True
        if not self._is_milvus_reachable():
            return
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
            if not client.has_collection(settings.milvus_collection):
                return
            docs = client.query(
                collection_name=settings.milvus_collection,
                filter="",
                limit=100,
                output_fields=[
                    "id", "doc_id", "parent_id", "title", "source",
                    "section", "tenant_id", "access_roles", "content", "parent_content",
                ],
            )
            if not docs:
                return
            embedder = ModelRegistry.get_embeddings()
            hydrated: List[ChunkRecord] = []
            for idx, meta in enumerate(docs):
                cid = meta.get("id") or f"chk_milvus_{idx}"
                raw_roles = meta.get("access_roles", "public")
                roles = [r.strip() for r in raw_roles.split(",") if r.strip()] if isinstance(raw_roles, str) else list(raw_roles)
                content = meta.get("content", "")
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
                        content=content,
                        parent_content=meta.get("parent_content", content),
                        chunk_index=idx,
                        token_estimate=max(1, len(content.split())),
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
            substage_latencies: Dict[str, float] = {}
            total_corpus = len(self._chunk_order)
            if not self._chunk_order:
                self._last_funnel = {
                    "total_corpus_chunks": 0,
                    "rbac_authorized_chunks": 0,
                    "dense_candidates_count": 0,
                    "sparse_candidates_count": 0,
                    "graph_candidates_count": 0,
                    "rrf_fused_pool_count": 0,
                    "parent_promoted_count": 0,
                    "reranked_pool_count": 0,
                    "crag_relevant_count": 0,
                    "mmr_selected_count": 0,
                    "pruned_chunks": [],
                    "substage_latencies_ms": {},
                }
                return []

            t_rbac = time.perf_counter()
            authorized_ids = self._authorized_chunk_ids(tenant_id, user_roles)
            substage_latencies["rbac_prefilter_ms"] = round((time.perf_counter() - t_rbac) * 1000.0, 3)
            if not authorized_ids:
                self._last_funnel = {
                    "total_corpus_chunks": total_corpus,
                    "rbac_authorized_chunks": 0,
                    "dense_candidates_count": 0,
                    "sparse_candidates_count": 0,
                    "graph_candidates_count": 0,
                    "rrf_fused_pool_count": 0,
                    "parent_promoted_count": 0,
                    "reranked_pool_count": 0,
                    "crag_relevant_count": 0,
                    "mmr_selected_count": 0,
                    "pruned_chunks": [],
                    "substage_latencies_ms": substage_latencies,
                }
                return []

            clean_queries = [q.strip() for q in queries if q and q.strip()]
            if not clean_queries:
                return []
            primary_query = clean_queries[0]

            query_terms_set = set(tokenize_content_words(" ".join(clean_queries)))
            query_entities_set = set()
            for q in clean_queries:
                query_entities_set.update(extract_entities(q))
                for tok in tokenize_content_words(q):
                    if "_" in tok or "-" in tok:
                        query_entities_set.add(tok)

            # 1. Dense Bi-Encoder Rankings (Pre-Filtered to authorized_ids)
            t_dense = time.perf_counter()
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
            substage_latencies["dense_search_ms"] = round((time.perf_counter() - t_dense) * 1000.0, 3)

            # 2. Sparse BM25 Rankings (Pre-Filtered to authorized_ids)
            t_bm25 = time.perf_counter()
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
            substage_latencies["bm25_search_ms"] = round((time.perf_counter() - t_bm25) * 1000.0, 3)

            # 3. Entity Knowledge Graph 1-Hop Expansion (for Tri-Modal GraphRAG)
            t_graph = time.perf_counter()
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
            substage_latencies["entity_graph_ms"] = round((time.perf_counter() - t_graph) * 1000.0, 3)

            # 4. Reciprocal Rank Fusion (RRF)
            t_rrf = time.perf_counter()
            candidate_ids = (
                set(dense_ranks_per_chunk.keys())
                | set(sparse_ranks_per_chunk.keys())
                | set(graph_ranks_per_chunk.keys())
            )
            if not candidate_ids and mode == "sparse":
                candidate_ids = set(list(authorized_ids)[:top_k])

            candidates: List[RetrievedChunk] = []
            rrf_k = settings.rrf_k
            now_ts = time.time()
            for cid in candidate_ids:
                record = self._chunks_by_id[cid]
                d_rank = dense_ranks_per_chunk.get(cid)
                d_score = dense_scores_per_chunk.get(cid)
                s_rank = sparse_ranks_per_chunk.get(cid)
                s_score = sparse_scores_per_chunk.get(cid)
                g_rank = graph_ranks_per_chunk.get(cid)
                g_score = graph_scores_per_chunk.get(cid)

                d_rrf = (dense_weight / (rrf_k + d_rank)) if (d_rank is not None and mode != "sparse") else 0.0
                s_rrf = (sparse_weight / (rrf_k + s_rank)) if (s_rank is not None and mode != "dense") else 0.0
                g_rrf = (graph_weight / (rrf_k + g_rank)) if (g_rank is not None and mode == "hybrid_graph_rerank") else 0.0

                # Reddit/Yelp freshness time-decay + Figma version precedence boost
                raw_updated_at = record.metadata.get("updated_at")
                updated_ts = float(raw_updated_at) if isinstance(raw_updated_at, (int, float)) else now_ts
                age_days = max(0.0, (now_ts - updated_ts) / 86400.0)
                freshness_boost = round(0.85 + 0.15 * math.exp(-0.004 * min(age_days, 365.0)), 4)
                version_num = max(1, int(record.metadata.get("version") or 1))
                version_boost = round(1.0 + 0.04 * min(version_num - 1, 10), 4)
                rrf_val = (d_rrf + s_rrf + g_rrf) * freshness_boost * version_boost

                initial_prob = 0.0
                if d_score is not None and s_score is not None:
                    initial_prob = min(0.95, max(0.0, 0.5 * d_score + 0.5 * min(1.0, s_score / 8.0)))
                elif d_score is not None:
                    initial_prob = max(0.0, min(0.95, d_score))
                elif s_score is not None:
                    initial_prob = max(0.0, min(0.95, s_score / 8.0))

                chunk_terms = set(tokenize_content_words(record.parent_content))
                matched_terms = sorted(query_terms_set & chunk_terms)
                chunk_entities = set(record.entities or extract_entities(record.parent_content))
                matched_entities = sorted(query_entities_set & chunk_entities)

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
                        matched_terms=matched_terms,
                        matched_entities=matched_entities,
                        rrf_breakdown={
                            "dense": round(d_rrf, 6),
                            "sparse": round(s_rrf, 6),
                            "graph": round(g_rrf, 6),
                        },
                        freshness_boost=freshness_boost,
                        version_boost=version_boost,
                    )
                )

            raw_candidates_count = len(candidates)
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
            deduped_candidates_count = len(candidates)
            substage_latencies["rrf_and_parent_promotion_ms"] = round((time.perf_counter() - t_rrf) * 1000.0, 3)

            # 6. Stage 2: Cross-Encoder Reranking + Negation Adjustment + MMR Diversity
            t_rerank = time.perf_counter()
            pruned_chunks: List[Dict[str, Any]] = []
            if mode in ("hybrid_rerank", "hybrid_graph_rerank"):
                pool = candidates[: max(top_k * 3, 12)]
                reranked = self._reranker.rerank(primary_query, pool, top_k=max(top_k * 2, top_k))
                reranked = context_postprocessor.apply_negation_adjustment(primary_query, reranked)
                final_chunks = context_postprocessor.select_mmr(reranked, top_k=top_k)
                selected_ids = {c.chunk_id for c in final_chunks}
                for cand in reranked:
                    if not cand.is_relevant:
                        pruned_chunks.append(
                            {
                                "chunk_id": cand.chunk_id,
                                "title": cand.title,
                                "calibrated_prob": cand.calibrated_prob,
                                "reason": "crag_below_threshold",
                            }
                        )
                    elif cand.chunk_id not in selected_ids:
                        pruned_chunks.append(
                            {
                                "chunk_id": cand.chunk_id,
                                "title": cand.title,
                                "calibrated_prob": cand.calibrated_prob,
                                "reason": "mmr_diversity_filtered",
                            }
                        )
            else:
                pool = candidates
                final_chunks = candidates[:top_k]
                for cand in final_chunks:
                    if not cand.is_relevant:
                        pruned_chunks.append(
                            {
                                "chunk_id": cand.chunk_id,
                                "title": cand.title,
                                "calibrated_prob": cand.calibrated_prob,
                                "reason": "below_threshold",
                            }
                        )
            substage_latencies["cross_encoder_and_mmr_ms"] = round((time.perf_counter() - t_rerank) * 1000.0, 3)

            self._last_funnel = {
                "total_corpus_chunks": total_corpus,
                "rbac_authorized_chunks": len(authorized_ids),
                "dense_candidates_count": len(dense_ranks_per_chunk),
                "sparse_candidates_count": len(sparse_ranks_per_chunk),
                "graph_candidates_count": len(graph_ranks_per_chunk),
                "rrf_fused_pool_count": raw_candidates_count,
                "parent_promoted_count": deduped_candidates_count,
                "reranked_pool_count": len(pool),
                "crag_relevant_count": sum(1 for c in final_chunks if c.is_relevant),
                "mmr_selected_count": len(final_chunks),
                "pruned_chunks": pruned_chunks[:5],
                "substage_latencies_ms": substage_latencies,
            }

            return final_chunks

    def find_doc_ids_by_external_id(self, external_id: str, tenant_id: str = "default") -> List[str]:
        """Finds all active doc_ids matching a CDC external_id within a tenant namespace."""
        with self._lock:
            return sorted(
                {
                    chk.doc_id
                    for chk in self._chunks_by_id.values()
                    if chk.tenant_id == tenant_id and chk.metadata.get("external_id") == external_id
                }
            )

    def delete_by_doc_id(self, doc_id: str) -> int:
        """GDPR Article 17 Surgical Erasure: removes all chunks of doc_id across Dense, BM25, and Entity Graph."""
        with self._lock:
            to_remove = {cid for cid, chk in self._chunks_by_id.items() if chk.doc_id == doc_id}
            if not to_remove:
                return 0
            for cid in to_remove:
                self._chunks_by_id.pop(cid, None)
                self._dense_vectors.pop(cid, None)
            self._chunk_order = [cid for cid in self._chunk_order if cid not in to_remove]
            entity_graph.remove_chunks(to_remove)
            self._rebuild_bm25()
            return len(to_remove)

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

    def list_documents(self) -> List[Dict[str, object]]:
        """Aggregates indexed chunks by doc_id for the Compliance & Document Inventory API."""
        with self._lock:
            grouped: Dict[str, Dict[str, object]] = {}
            for cid in self._chunk_order:
                chk = self._chunks_by_id[cid]
                if chk.doc_id not in grouped:
                    grouped[chk.doc_id] = {
                        "doc_id": chk.doc_id,
                        "title": chk.title,
                        "source": chk.source,
                        "tenant_id": chk.tenant_id,
                        "access_roles": list(chk.access_roles),
                        "chunk_count": 0,
                        "parent_ids": set(),
                        "sample_chunk_ids": [],
                    }
                entry = grouped[chk.doc_id]
                entry["chunk_count"] = int(entry["chunk_count"]) + 1  # type: ignore[arg-type]
                entry["parent_ids"].add(chk.parent_id)  # type: ignore[attr-defined]
                sample_list: List[str] = entry["sample_chunk_ids"]  # type: ignore[assignment]
                if len(sample_list) < 3:
                    sample_list.append(chk.chunk_id)

            documents: List[Dict[str, object]] = []
            for doc in grouped.values():
                parent_ids = doc.pop("parent_ids")
                doc["parent_windows_count"] = len(parent_ids)  # type: ignore[arg-type]
                documents.append(doc)
            return documents

    def clear(self) -> None:
        with self._lock:
            self._chunks_by_id.clear()
            self._chunk_order.clear()
            self._dense_vectors.clear()
            self._bm25 = None
            entity_graph.clear()


retrieval_engine = HybridRetrievalEngine()
