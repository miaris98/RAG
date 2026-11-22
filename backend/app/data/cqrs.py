"""P4.3 CQRS Materialized Read Model Projection & Database-per-Service Ownership Contracts."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List


DATABASE_PER_SERVICE_BOUNDARIES: List[Dict[str, Any]] = [
    {
        "service": "retrieval_service",
        "owned_datastore": "Milvus Vector DB + BM25 Lexical Index + Entity Knowledge Graph",
        "access_mode": "API-only via HybridRetrievalEngine / POST /ingest / POST /retrieve",
        "shared_db_forbidden": True,
    },
    {
        "service": "sql_opsbot_service",
        "owned_datastore": "Operational SQLite (service_inventory, active_incidents)",
        "access_mode": "Read-only SELECT API via GET /inventory/sql-status & POST /inventory/sql-query",
        "shared_db_forbidden": True,
    },
    {
        "service": "compliance_audit_service",
        "owned_datastore": "Append-only SHA-256 Hash-Chained Audit Ledger",
        "access_mode": "Append-only via TamperEvidentAuditLogger / GET /audit-logs & /compliance/audit-report",
        "shared_db_forbidden": True,
    },
    {
        "service": "feedback_service",
        "owned_datastore": "User Feedback Satisfaction Store",
        "access_mode": "API-only via POST /feedback & GET /feedback",
        "shared_db_forbidden": True,
    },
]


class CQRSReadProjection:
    """Lock-free snapshot read model updated by write commands (ingest, CDC, GDPR erasure)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._docs: Dict[str, Dict[str, Any]] = {}
        self._version: int = 0
        self._last_updated: float = 0.0
        self._materialized_snapshot: Dict[str, Any] = self._build_snapshot_unlocked()

    def _build_snapshot_unlocked(self) -> Dict[str, Any]:
        by_tenant: Dict[str, int] = {}
        by_role: Dict[str, int] = {}
        total_chunks = 0
        for doc in self._docs.values():
            t = doc["tenant_id"]
            by_tenant[t] = by_tenant.get(t, 0) + 1
            for role in doc["access_roles"]:
                by_role[role] = by_role.get(role, 0) + 1
            total_chunks += int(doc.get("chunk_count", 0))
        return {
            "projection_version": self._version,
            "last_projected_at": round(self._last_updated, 3),
            "total_documents": len(self._docs),
            "total_chunks": total_chunks,
            "documents_by_tenant": by_tenant,
            "documents_by_role": by_role,
            "documents": list(self._docs.values()),
        }

    def project_document_upserted(
        self,
        doc_id: str,
        title: str,
        source: str,
        tenant_id: str,
        access_roles: List[str],
        chunk_count: int,
    ) -> None:
        if not doc_id:
            return
        with self._lock:
            self._docs[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "tenant_id": tenant_id,
                "access_roles": list(access_roles),
                "chunk_count": int(chunk_count),
                "projected_at": round(time.time(), 3),
            }
            self._version += 1
            self._last_updated = time.time()
            self._materialized_snapshot = self._build_snapshot_unlocked()

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            existed = self._docs.pop(doc_id, None) is not None
            if existed:
                self._version += 1
                self._last_updated = time.time()
                self._materialized_snapshot = self._build_snapshot_unlocked()
            return existed

    def snapshot(self) -> Dict[str, Any]:
        """Returns the pre-materialized read projection in O(1) without scanning write stores."""
        return dict(self._materialized_snapshot)

    def clear(self) -> None:
        with self._lock:
            self._docs.clear()
            self._version = 0
            self._last_updated = 0.0
            self._materialized_snapshot = self._build_snapshot_unlocked()


cqrs_read_model = CQRSReadProjection()
