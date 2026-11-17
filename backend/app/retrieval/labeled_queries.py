"""Vetted Golden 'Labeled Queries' Sidecar Store (inspired by Block & Instacart)."""

import threading
from typing import Any, Dict, List, Optional


class LabeledQueryStore:
    """Priority sidecar cache of human-vetted query-answer pairs scoped by RBAC roles."""

    def __init__(self):
        self._lock = threading.RLock()
        self._entries: List[Dict[str, Any]] = []

    def add_vetted_query(
        self,
        query: str,
        vetted_answer: str,
        access_roles: Optional[List[str]] = None,
        tenant_id: str = "default",
    ) -> None:
        with self._lock:
            self._entries.append(
                {
                    "query_norm": query.strip().lower(),
                    "query": query.strip(),
                    "answer": vetted_answer,
                    "access_roles": {r.strip().lower() for r in (access_roles or ["public"])},
                    "tenant_id": tenant_id,
                }
            )

    def lookup(
        self,
        query: str,
        tenant_id: str = "default",
        user_roles: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        norm = query.strip().lower()
        caller_roles = {r.strip().lower() for r in (user_roles or ["public"])}
        with self._lock:
            for item in self._entries:
                if item["tenant_id"] != tenant_id:
                    continue
                if not (item["access_roles"] & caller_roles):
                    continue
                if item["query_norm"] == norm:
                    return dict(item)
        return None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


labeled_query_store = LabeledQueryStore()
