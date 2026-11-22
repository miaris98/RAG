"""Zendesk-inspired User Feedback Loop store linked to trace_id, retrieved chunks, and RBAC roles."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Literal, Optional


class FeedbackStore:
    """Thread-safe in-memory user feedback store for thumbs-up/down satisfaction telemetry."""

    def __init__(self, max_items: int = 500):
        self._lock = threading.Lock()
        self._items: List[Dict[str, Any]] = []
        self._max_items = max_items

    def record(
        self,
        rating: Literal["up", "down"],
        query: str,
        trace_id: Optional[str] = None,
        chunk_ids: Optional[List[str]] = None,
        user_roles: Optional[List[str]] = None,
        comment: str = "",
    ) -> Dict[str, Any]:
        entry = {
            "feedback_id": f"fb_{int(time.time() * 1000)}_{len(self._items) + 1}",
            "timestamp": round(time.time(), 3),
            "rating": rating,
            "query": query[:300],
            "trace_id": trace_id,
            "chunk_ids": list(chunk_ids or [])[:10],
            "user_roles": list(user_roles or ["public"]),
            "comment": (comment or "")[:500],
        }
        with self._lock:
            self._items.append(entry)
            if len(self._items) > self._max_items:
                self._items = self._items[-self._max_items :]
        return entry

    def summary(self, limit: int = 25) -> Dict[str, Any]:
        with self._lock:
            total = len(self._items)
            up = sum(1 for item in self._items if item["rating"] == "up")
            down = total - up
            rate = round(up / total, 4) if total > 0 else 1.0
            recent = list(reversed(self._items[-max(1, limit) :]))
        return {
            "total_feedback": total,
            "thumbs_up": up,
            "thumbs_down": down,
            "satisfaction_rate": rate,
            "recent_feedback": recent,
        }

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


feedback_store = FeedbackStore()
