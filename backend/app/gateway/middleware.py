"""P4.2 API Gateway Middleware: Request-ID propagation, Security Headers, and Token-Bucket Rate Limiting."""

from __future__ import annotations

from collections import deque
import math
import threading
import time
import uuid
from typing import Deque, Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.resilience.progressive_delivery import feature_flags


class GatewayRateLimiter:
    """Sliding-window per-client/tenant rate limiter returning (allowed, retry_after_sec)."""

    def __init__(self, max_requests_per_window: int = 600, window_sec: float = 60.0):
        self.max_requests_per_window = max(1, int(max_requests_per_window))
        self.window_sec = max(1.0, float(window_sec))
        self._lock = threading.Lock()
        self._buckets: Dict[str, Deque[float]] = {}
        self._total_rejected = 0

    def allow(self, client_key: str) -> Tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_sec
        with self._lock:
            dq = self._buckets.setdefault(client_key, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests_per_window:
                self._total_rejected += 1
                retry_after = max(1, int(math.ceil(self.window_sec - (now - dq[0]))))
                return False, retry_after
            dq.append(now)
            return True, 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "active_clients": len(self._buckets),
                "max_requests_per_window": self.max_requests_per_window,
                "total_rejected": self._total_rejected,
            }


gateway_rate_limiter = GatewayRateLimiter(max_requests_per_window=600, window_sec=60.0)


class APIGatewayMiddleware(BaseHTTPMiddleware):
    """Cross-cutting API Gateway layer enforcing rate limits, request correlation IDs, and security headers."""

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:12]}"
        client_key = (
            request.headers.get("X-Tenant-Id")
            or request.headers.get("X-Peer-Agent-Id")
            or (request.client.host if request.client else "local")
        )
        allowed, retry_after = gateway_rate_limiter.allow(client_key)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "API Gateway rate limit exceeded.", "retry_after_sec": retry_after},
                headers={
                    "Retry-After": str(retry_after),
                    "X-Request-Id": req_id,
                    "X-Gateway-Service": "staff-rag-gateway",
                },
            )

        response = await call_next(request)
        response.headers["X-Request-Id"] = req_id
        response.headers["X-Gateway-Service"] = "staff-rag-gateway"
        response.headers["X-Release-Track"] = feature_flags.active_release_track
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        return response
