"""Lightweight Tri-Modal Entity Knowledge Graph for GraphRAG 1-Hop Expansion.

Extracts technical entities (error codes like ERR_LEDGER_9042, service names like AegisPay/Milvus/MinIO,
rotation IDs like PD-EMEA-8821, ciphers like AES-256-GCM, and capitalized domain concepts) at ingestion
and builds a bipartite Chunk <-> Entity + Entity <-> Entity co-occurrence graph.
"""

import re
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


TECHNICAL_CODE_PATTERN = re.compile(r"\b[A-Z]{2,}(?:_[A-Z0-9]+)+|\b[A-Z]{2,}-[A-Z0-9-]{2,}\b")
CAMEL_OR_PROPER_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,})?)\b")

IGNORED_COMMON_CAPS = {
    "Document", "Source", "Section", "Overview", "Critical", "Internal",
    "External", "According", "Comparison", "General", "FixedChunk", "Persisted",
    "Untitled", "Main", "Because", "When", "What", "Which", "Where", "This",
}


def extract_entities(text: str) -> List[str]:
    """Extracts normalized technical entities and identifiers from text."""
    entities: Set[str] = set()

    for match in TECHNICAL_CODE_PATTERN.findall(text):
        entities.add(match.lower())

    for match in CAMEL_OR_PROPER_PATTERN.findall(text):
        if match in IGNORED_COMMON_CAPS:
            continue
        cleaned = match.strip().lower()
        if len(cleaned) >= 4:
            entities.add(cleaned)

    return sorted(entities)


class EntityKnowledgeGraph:
    """In-memory bipartite & entity-cooccurrence graph for 1-hop GraphRAG retrieval."""

    def __init__(self):
        self._lock = threading.RLock()
        self._entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)
        self._chunk_to_entities: Dict[str, Set[str]] = defaultdict(set)
        self._entity_neighbors: Dict[str, Set[str]] = defaultdict(set)

    def add_chunk(self, chunk_id: str, text: str) -> List[str]:
        """Extracts entities from chunk text and registers 1-hop co-occurrence edges."""
        ents = extract_entities(text)
        with self._lock:
            for ent in ents:
                self._entity_to_chunks[ent].add(chunk_id)
                self._chunk_to_entities[chunk_id].add(ent)
            for i, e1 in enumerate(ents):
                for e2 in ents[i + 1 :]:
                    self._entity_neighbors[e1].add(e2)
                    self._entity_neighbors[e2].add(e1)
        return ents

    def score_chunks(
        self,
        queries: List[str],
        allowed_chunk_ids: Optional[Set[str]] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Scores chunks by direct entity match (1.0) + 1-hop neighbor entity propagation (0.35)."""
        query_entities: Set[str] = set()
        for q in queries:
            query_entities.update(extract_entities(q))
            # Also match lowercase technical tokens directly against known graph entities
            for tok in re.findall(r"[a-z0-9_.-]{3,}", q.lower()):
                if tok in self._entity_to_chunks:
                    query_entities.add(tok)

        if not query_entities:
            return []

        with self._lock:
            chunk_scores: Dict[str, float] = defaultdict(float)

            for q_ent in query_entities:
                # Direct entity -> chunk hits
                for cid in self._entity_to_chunks.get(q_ent, ()):
                    if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
                        continue
                    chunk_scores[cid] += 1.0

                # 1-hop entity -> neighbor entity -> chunk hits
                for neighbor in self._entity_neighbors.get(q_ent, ()):
                    for cid in self._entity_to_chunks.get(neighbor, ()):
                        if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
                            continue
                        chunk_scores[cid] += 0.35

            ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "total_entities": len(self._entity_to_chunks),
                "total_linked_chunks": len(self._chunk_to_entities),
            }

    def clear(self) -> None:
        with self._lock:
            self._entity_to_chunks.clear()
            self._chunk_to_entities.clear()
            self._entity_neighbors.clear()


entity_graph = EntityKnowledgeGraph()
