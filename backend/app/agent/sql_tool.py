"""#25 Agoda OpsBot Structured SQL Live Inventory Tool (Read-Only SQLite alongside Vector RAG)."""

from __future__ import annotations

import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional


FORBIDDEN_SQL_RE = re.compile(
    r"\b(?:DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|ATTACH|DETACH|PRAGMA|REPLACE|VACUUM|TRIGGER)\b",
    re.IGNORECASE,
)
LIVE_SQL_TRIGGER_RE = re.compile(
    r"\b(?:live\s+(?:status|inventory|service|incident)|service\s+inventory|active\s+incidents?|"
    r"active\s+nodes|error\s+budget|opsbot|sql\s+status|cluster\s+status)\b",
    re.IGNORECASE,
)


class LiveSQLInventoryTool:
    """Thread-safe read-only SQLite live inventory & incident store fused with vector RAG."""

    def __init__(self):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._seed_tables()

    def _seed_tables(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS service_inventory (
                    service_name TEXT PRIMARY KEY,
                    region TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active_nodes INTEGER NOT NULL,
                    p95_latency_ms REAL NOT NULL,
                    error_budget_pct REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_incidents (
                    incident_id TEXT PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    error_code TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    owner_rotation TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                INSERT OR REPLACE INTO service_inventory VALUES
                    ('aegispay-ledger', 'eu-central-1', 'degraded', 12, 64.2, 82.5),
                    ('milvus-vector-cluster', 'eu-central-1', 'healthy', 6, 18.4, 99.4),
                    ('rabbitmq-worker-pool', 'eu-central-1', 'healthy', 8, 11.2, 99.8),
                    ('rag-gateway-api', 'eu-west-1', 'healthy', 4, 41.5, 98.9);
                INSERT OR REPLACE INTO active_incidents VALUES
                    ('INC-9042', 'aegispay-ledger', 'ERR_LEDGER_9042', 'SEV1', 'PD-EMEA-8821', 'investigating'),
                    ('INC-7719', 'rabbitmq-worker-pool', 'ERR_VAULT_7719', 'SEV2', 'PD-SEC-4410', 'mitigated');
                """
            )
            self._conn.commit()

    def execute_readonly_sql(self, sql: str) -> Dict[str, Any]:
        """Executes a strictly validated read-only SELECT statement."""
        cleaned = (sql or "").strip()
        if not cleaned:
            raise ValueError("SQL query cannot be empty.")
        if ";" in cleaned or "--" in cleaned or "/*" in cleaned:
            raise ValueError("Multi-statement SQL and inline SQL comments are forbidden.")
        if not cleaned.lower().startswith("select "):
            raise ValueError("Only read-only SELECT statements are permitted.")
        if FORBIDDEN_SQL_RE.search(cleaned):
            raise ValueError("Destructive or schema-modifying SQL keywords are blocked.")

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(cleaned)
            rows = [dict(r) for r in cur.fetchall()[:100]]
            cols = [desc[0] for desc in (cur.description or [])]
        return {
            "sql": cleaned,
            "columns": cols,
            "row_count": len(rows),
            "rows": rows,
        }

    def get_live_snapshot(self) -> Dict[str, Any]:
        services = self.execute_readonly_sql(
            "SELECT service_name, region, status, active_nodes, p95_latency_ms, error_budget_pct "
            "FROM service_inventory ORDER BY service_name"
        )["rows"]
        incidents = self.execute_readonly_sql(
            "SELECT incident_id, service_name, error_code, severity, owner_rotation, status "
            "FROM active_incidents ORDER BY severity"
        )["rows"]
        return {"services": services, "active_incidents": incidents}

    def maybe_answer_from_live_sql(self, query: str) -> Optional[Dict[str, Any]]:
        """Detects live ops/inventory questions and returns structured SQL evidence alongside RAG."""
        if not query or not LIVE_SQL_TRIGGER_RE.search(query):
            return None

        snap = self.get_live_snapshot()
        lines = [
            "### Live SQL Inventory & Incident DB (Agoda OpsBot Tool)",
            "| Service | Region | Status | Active Nodes | p95 Latency | Error Budget |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
        for svc in snap["services"]:
            lines.append(
                f"| `{svc['service_name']}` | `{svc['region']}` | `{svc['status']}` | "
                f"`{svc['active_nodes']}` | `{svc['p95_latency_ms']} ms` | `{svc['error_budget_pct']}%` |"
            )
        lines.append("")
        lines.append("| Incident ID | Service | Error Code | Severity | Rotation | Status |")
        lines.append("| :--- | :--- | :--- | :---: | :--- | :---: |")
        for inc in snap["active_incidents"]:
            lines.append(
                f"| `{inc['incident_id']}` | `{inc['service_name']}` | `{inc['error_code']}` | "
                f"`{inc['severity']}` | `{inc['owner_rotation']}` | `{inc['status']}` |"
            )

        return {
            "markdown": "\n".join(lines),
            "snapshot": snap,
        }


sql_inventory_tool = LiveSQLInventoryTool()
