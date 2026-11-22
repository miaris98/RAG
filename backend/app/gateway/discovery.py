"""P4.2 Dynamic Service Discovery Registry (Consul / Kubernetes DNS-compatible heartbeat & round-robin)."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class ServiceRegistry:
    """Thread-safe dynamic service registry with TTL heartbeats, stale eviction, and round-robin resolution."""

    def __init__(self, seed_defaults: bool = True):
        self._lock = threading.RLock()
        self._services: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._rr_cursors: Dict[str, int] = {}
        if seed_defaults:
            self._seed_local_services()

    def _seed_local_services(self) -> None:
        self.register("rag-gateway-api", "gw-local-1", "127.0.0.1", 8000, ttl_sec=86400, version="2.0.0")
        self.register("hybrid-retrieval-engine", "ret-local-1", "127.0.0.1", 8000, ttl_sec=86400, version="2.0.0")
        self.register("sql-opsbot-db", "sql-local-1", "127.0.0.1", 8000, ttl_sec=86400, version="2.0.0")
        self.register("a2a-supervisor", "a2a-local-1", "127.0.0.1", 8000, ttl_sec=86400, version="2.0.0")

    def register(
        self,
        service_name: str,
        instance_id: str,
        host: str,
        port: int,
        ttl_sec: float = 30.0,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.time()
        record = {
            "service_name": service_name,
            "instance_id": instance_id,
            "host": host,
            "port": int(port),
            "version": version,
            "ttl_sec": float(ttl_sec),
            "registered_at": round(now, 3),
            "last_heartbeat": round(now, 3),
            "expires_at": round(now + float(ttl_sec), 3),
            "healthy": True,
            "metadata": dict(metadata or {}),
        }
        with self._lock:
            self._services.setdefault(service_name, {})[instance_id] = record
        return dict(record)

    def heartbeat(self, service_name: str, instance_id: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            inst = self._services.get(service_name, {}).get(instance_id)
            if not inst:
                return None
            inst["last_heartbeat"] = round(now, 3)
            inst["expires_at"] = round(now + float(inst["ttl_sec"]), 3)
            inst["healthy"] = True
            return dict(inst)

    def prune_expired(self) -> int:
        now = time.time()
        evicted = 0
        with self._lock:
            empty_services: List[str] = []
            for svc_name, instances in self._services.items():
                expired_ids = [iid for iid, rec in instances.items() if now > float(rec["expires_at"])]
                for iid in expired_ids:
                    instances.pop(iid, None)
                    evicted += 1
                if not instances:
                    empty_services.append(svc_name)
            for svc_name in empty_services:
                self._services.pop(svc_name, None)
        return evicted

    def resolve(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Round-robins across currently healthy, non-expired instances of service_name."""
        self.prune_expired()
        with self._lock:
            instances = [rec for rec in self._services.get(service_name, {}).values() if rec.get("healthy")]
            if not instances:
                return None
            instances.sort(key=lambda r: r["instance_id"])
            cursor = self._rr_cursors.get(service_name, 0) % len(instances)
            self._rr_cursors[service_name] = cursor + 1
            return dict(instances[cursor])

    def list_services(self) -> Dict[str, Any]:
        evicted = self.prune_expired()
        with self._lock:
            snapshot = {
                svc: [dict(rec) for rec in sorted(insts.values(), key=lambda r: r["instance_id"])]
                for svc, insts in sorted(self._services.items())
            }
            return {
                "service_count": len(snapshot),
                "total_instances": sum(len(v) for v in snapshot.values()),
                "evicted_expired": evicted,
                "services": snapshot,
            }


service_registry = ServiceRegistry(seed_defaults=True)
