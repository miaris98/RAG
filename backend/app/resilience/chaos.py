"""P4.5 Gated Chaos Engineering Fault Injector for verifying Circuit Breakers, Bulkheads & Degradation."""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable, Dict, Optional


ALLOWED_CHAOS_TARGETS = {"ollama", "duckduckgo", "milvus", "rabbitmq", "sql_opsbot"}
ALLOWED_CHAOS_MODES = {"failure", "latency"}


class ChaosInjectedError(RuntimeError):
    """Raised intentionally by ChaosController to validate fault-tolerance mechanisms."""


class ChaosController:
    """Thread-safe, explicitly gated chaos fault injector (off by default)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._enabled: bool = False
        self._target: str = "ollama"
        self._mode: str = "failure"
        self._failure_rate: float = 0.0
        self._latency_ms: int = 0
        self._total_faults_injected: int = 0
        self._total_latencies_injected: int = 0

    def configure(
        self,
        enabled: bool,
        target: str = "ollama",
        mode: str = "failure",
        failure_rate: float = 0.0,
        latency_ms: int = 0,
    ) -> Dict[str, Any]:
        norm_target = target.strip().lower()
        norm_mode = mode.strip().lower()
        if norm_target not in ALLOWED_CHAOS_TARGETS:
            raise ValueError(f"Unsupported chaos target '{target}'. Allowed: {sorted(ALLOWED_CHAOS_TARGETS)}")
        if norm_mode not in ALLOWED_CHAOS_MODES:
            raise ValueError(f"Unsupported chaos mode '{mode}'. Allowed: {sorted(ALLOWED_CHAOS_MODES)}")
        rate = float(failure_rate)
        if rate < 0.0 or rate > 1.0:
            raise ValueError("failure_rate must be between 0.0 and 1.0.")
        lat = int(latency_ms)
        if lat < 0 or lat > 10000:
            raise ValueError("latency_ms must be between 0 and 10000.")

        with self._lock:
            self._enabled = bool(enabled)
            self._target = norm_target
            self._mode = norm_mode
            self._failure_rate = rate
            self._latency_ms = lat
        return self.snapshot()

    def reset(self) -> Dict[str, Any]:
        with self._lock:
            self._enabled = False
            self._failure_rate = 0.0
            self._latency_ms = 0
        return self.snapshot()

    def maybe_inject(
        self,
        target: str,
        sleep_fn: Callable[[float], None] = time.sleep,
        rng_fn: Callable[[], float] = random.random,
    ) -> Optional[str]:
        """Injects configured synthetic latency or failure if chaos is enabled for target."""
        with self._lock:
            if not self._enabled or self._target != target.strip().lower():
                return None
            mode = self._mode
            rate = self._failure_rate
            lat_ms = self._latency_ms

        if mode == "latency" and lat_ms > 0:
            with self._lock:
                self._total_latencies_injected += 1
            sleep_fn(lat_ms / 1000.0)
            return f"latency:{lat_ms}ms"

        if mode == "failure" and rate > 0.0:
            if rate >= 1.0 or rng_fn() < rate:
                with self._lock:
                    self._total_faults_injected += 1
                raise ChaosInjectedError(f"Chaos fault injected on downstream target '{target}'.")
        return None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "target": self._target,
                "mode": self._mode,
                "failure_rate": self._failure_rate,
                "latency_ms": self._latency_ms,
                "total_faults_injected": self._total_faults_injected,
                "total_latencies_injected": self._total_latencies_injected,
                "allowed_targets": sorted(ALLOWED_CHAOS_TARGETS),
                "allowed_modes": sorted(ALLOWED_CHAOS_MODES),
            }


chaos_controller = ChaosController()
