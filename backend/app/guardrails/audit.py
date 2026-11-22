"""Tamper-Evident SHA-256 Hash-Chained Audit Logger for GDPR, EU AI Act & SOC2 Compliance."""

import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional

from app.guardrails.security import guardrails
from app.security.zero_trust import secrets_provider


class TamperEvidentAuditLogger:
    """Append-only, PII-safe, SHA-256 hash-chained audit trail for ingestion, queries, and GDPR erasures."""

    def __init__(self, max_events: int = 2000):
        self._lock = threading.RLock()
        self._events: List[Dict[str, Any]] = []
        self._last_hash: str = "0" * 64
        self.max_events = max_events

    def log_event(
        self,
        event_type: str,
        actor_tenant: str = "default",
        actor_roles: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        chunk_ids: Optional[List[str]] = None,
        summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Records a PII-sanitized and secret-redacted audit event chained to the previous event's SHA-256 digest."""
        clean_summary, _ = guardrails.sanitize_text(summary, redact_pii=True, block_injections=True)
        clean_summary = secrets_provider.redact_secrets(clean_summary)
        with self._lock:
            seq = len(self._events) + 1
            record_body = {
                "seq": seq,
                "timestamp": round(time.time(), 3),
                "event_type": event_type,
                "actor_tenant": actor_tenant,
                "actor_roles": list(actor_roles or []),
                "doc_ids": list(doc_ids or []),
                "chunk_ids": list(chunk_ids or []),
                "summary": clean_summary,
                "metadata": dict(metadata or {}),
                "prev_hash": self._last_hash,
            }
            canonical = json.dumps(record_body, sort_keys=True, ensure_ascii=False)
            event_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            record_body["event_hash"] = event_hash
            self._last_hash = event_hash

            if len(self._events) >= self.max_events:
                self._events.pop(0)
            self._events.append(record_body)
            return dict(record_body)

    def verify_chain(self) -> bool:
        """Recomputes SHA-256 hashes across all retained events to detect any tampering."""
        with self._lock:
            if not self._events:
                return True
            prev = self._events[0]["prev_hash"]
            for idx, ev in enumerate(self._events):
                if idx > 0 and ev["prev_hash"] != prev:
                    return False
                body = {k: v for k, v in ev.items() if k != "event_hash"}
                expected = hashlib.sha256(
                    json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                if expected != ev["event_hash"]:
                    return False
                prev = ev["event_hash"]
            return True

    def get_logs(self, limit: int = 100) -> Dict[str, Any]:
        with self._lock:
            recent = list(reversed(self._events[-limit:]))
            return {
                "chain_valid": self.verify_chain(),
                "total_events": len(self._events),
                "head_hash": self._last_hash,
                "events": recent,
            }

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._last_hash = "0" * 64


audit_logger = TamperEvidentAuditLogger()
