"""P4.5 Progressive Delivery: Deterministic Canary Feature Flags & Blue-Green Release Track Manager."""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, Optional


VALID_RELEASE_TRACKS = {"blue", "green", "canary"}


class FeatureFlagManager:
    """Thread-safe feature flag & progressive delivery controller (Canary % + Blue-Green track)."""

    def __init__(self, initial_track: str = "blue"):
        self._lock = threading.Lock()
        self._active_release_track = initial_track if initial_track in VALID_RELEASE_TRACKS else "blue"
        self._flags: Dict[str, Dict[str, Any]] = {
            "hyde_v2_canary": {"enabled": True, "rollout_percentage": 5, "description": "Canary HyDE v2 expansion"},
            "rag_canary_reranker": {"enabled": True, "rollout_percentage": 10, "description": "Canary Cross-Encoder v2"},
            "late_chunking_default": {"enabled": True, "rollout_percentage": 100, "description": "Perplexity Late Chunking"},
        }

    @property
    def active_release_track(self) -> str:
        with self._lock:
            return self._active_release_track

    def set_release_track(self, track: str) -> str:
        normalized = track.strip().lower()
        if normalized not in VALID_RELEASE_TRACKS:
            raise ValueError(f"Release track must be one of {sorted(VALID_RELEASE_TRACKS)}, got '{track}'.")
        with self._lock:
            self._active_release_track = normalized
            return self._active_release_track

    def set_flag(
        self,
        flag_name: str,
        enabled: bool,
        rollout_percentage: int = 100,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        pct = int(rollout_percentage)
        if pct < 0 or pct > 100:
            raise ValueError("rollout_percentage must be between 0 and 100.")
        with self._lock:
            existing = self._flags.get(flag_name, {})
            entry = {
                "enabled": bool(enabled),
                "rollout_percentage": pct,
                "description": description if description is not None else existing.get("description", ""),
            }
            self._flags[flag_name] = entry
            return dict(entry)

    def evaluate(self, flag_name: str, subject_key: str = "default") -> bool:
        """Deterministically evaluates whether subject_key falls in the canary percentage bucket."""
        with self._lock:
            flag = self._flags.get(flag_name)
        if not flag or not flag["enabled"]:
            return False
        pct = int(flag["rollout_percentage"])
        if pct <= 0:
            return False
        if pct >= 100:
            return True
        digest = hashlib.sha256(f"{flag_name}:{subject_key}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < pct

    def snapshot(self, subject_key: str = "default") -> Dict[str, Any]:
        with self._lock:
            flags_copy = {k: dict(v) for k, v in self._flags.items()}
            track = self._active_release_track
        evals = {k: self.evaluate(k, subject_key=subject_key) for k in flags_copy}
        return {
            "active_release_track": track,
            "subject_key": subject_key,
            "flags": flags_copy,
            "evaluations": evals,
        }


feature_flags = FeatureFlagManager(initial_track="blue")
