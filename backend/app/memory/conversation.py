"""Session-scoped Conversational Memory & Coreference Contextualizer."""

import re
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, List

from app.retrieval.entity_graph import extract_entities


PRONOUN_PATTERN = re.compile(r"\b(it|its|they|their|them|this|that|those|the same|that error)\b", re.IGNORECASE)


class ConversationMemoryStore:
    """Thread-safe sliding-window turn & entity memory keyed by session_id."""

    def __init__(self, max_turns: int = 6):
        self._lock = threading.RLock()
        self._sessions: Dict[str, Deque[Dict[str, str]]] = defaultdict(lambda: deque(maxlen=max_turns))
        self._session_entities: Dict[str, List[str]] = defaultdict(list)

    def resolve_query(self, session_id: str, query: str) -> str:
        """Resolves pronouns in follow-up queries using the most recent salient entities in the session."""
        if not session_id or not query:
            return query
        with self._lock:
            entities = self._session_entities.get(session_id) or []
            if not entities:
                return query
            if PRONOUN_PATTERN.search(query):
                anchor_entities = ", ".join(entities[-3:])
                return f"{query} (context: {anchor_entities})"
            return query

    def record_turn(self, session_id: str, query: str, answer: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions[session_id].append({"query": query, "answer": answer[:400]})
            # Prioritize entities explicitly mentioned in the user's query as most salient
            query_ents = extract_entities(query)
            answer_ents = extract_entities(answer[:500])
            existing = self._session_entities[session_id]
            for ent in answer_ents + query_ents:
                if ent in existing:
                    existing.remove(ent)
                existing.append(ent)
            self._session_entities[session_id] = existing[-10:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._session_entities.clear()


conversation_memory = ConversationMemoryStore()
