"""P4.4 Zero Trust Short-Lived Service JWT Authority (HS256 + jti replay protection) & Least-Privilege Secrets."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set


MAX_SHORT_LIVED_TTL_SEC = 300.0
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b")
SECRET_PATTERN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,})\b"
)


class ZeroTrustError(PermissionError):
    """Raised when a Zero Trust service JWT fails signature, expiry, audience, or replay checks."""


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(val: str) -> bytes:
    pad = "=" * (-len(val) % 4)
    return base64.urlsafe_b64decode(val + pad)


class ZeroTrustTokenAuthority:
    """Issues and verifies short-lived HS256 service-to-service JWTs with jti replay protection."""

    def __init__(self, signing_secret: Optional[str] = None, issuer: str = "staff-rag-zero-trust"):
        self._secret = (signing_secret or os.getenv("ZERO_TRUST_JWT_SECRET") or "local-dev-zero-trust-signing-key").encode("utf-8")
        self.issuer = issuer
        self._lock = threading.Lock()
        self._seen_jti: Dict[str, float] = {}

    def _prune_jti_unlocked(self, now: float) -> None:
        expired = [jti for jti, exp_ts in self._seen_jti.items() if exp_ts < now]
        for jti in expired:
            self._seen_jti.pop(jti, None)

    def issue_token(
        self,
        subject: str,
        audience: str = "rag-gateway-api",
        roles: Optional[List[str]] = None,
        ttl_sec: float = 120.0,
    ) -> str:
        if ttl_sec <= 0 or ttl_sec > MAX_SHORT_LIVED_TTL_SEC:
            raise ValueError(f"Zero Trust service JWT ttl_sec must be in (0, {MAX_SHORT_LIVED_TTL_SEC}].")
        now = time.time()
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self.issuer,
            "sub": subject.strip(),
            "aud": audience.strip(),
            "roles": sorted({r.strip().lower() for r in (roles or ["public"]) if r and r.strip()}),
            "iat": round(now, 3),
            "exp": round(now + float(ttl_sec), 3),
            "jti": uuid.uuid4().hex,
        }
        h_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        p_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")
        sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        return f"{h_b64}.{p_b64}.{_b64url_encode(sig)}"

    def verify_token(
        self,
        token: str,
        expected_audience: str = "rag-gateway-api",
        prevent_replay: bool = True,
    ) -> Dict[str, Any]:
        if not token or token.count(".") != 2:
            raise ZeroTrustError("Malformed JWT token.")
        h_b64, p_b64, s_b64 = token.split(".")
        signing_input = f"{h_b64}.{p_b64}".encode("ascii")
        expected_sig = _b64url_encode(hmac.new(self._secret, signing_input, hashlib.sha256).digest())
        if not hmac.compare_digest(expected_sig, s_b64):
            raise ZeroTrustError("Invalid JWT cryptographic signature.")

        try:
            claims = json.loads(_b64url_decode(p_b64).decode("utf-8"))
        except Exception as exc:
            raise ZeroTrustError("Invalid JWT payload encoding.") from exc

        now = time.time()
        if claims.get("iss") != self.issuer:
            raise ZeroTrustError("Untrusted JWT issuer.")
        if expected_audience and claims.get("aud") != expected_audience:
            raise ZeroTrustError(f"JWT audience mismatch (expected '{expected_audience}').")
        exp = float(claims.get("exp") or 0.0)
        iat = float(claims.get("iat") or 0.0)
        if now > exp:
            raise ZeroTrustError("Service JWT has expired.")
        if (exp - iat) > MAX_SHORT_LIVED_TTL_SEC + 1.0:
            raise ZeroTrustError("Service JWT TTL exceeds short-lived Zero Trust policy (300s).")

        jti = str(claims.get("jti") or "")
        if prevent_replay and jti:
            with self._lock:
                self._prune_jti_unlocked(now)
                if jti in self._seen_jti:
                    raise ZeroTrustError("Service JWT jti replay detected and rejected.")
                self._seen_jti[jti] = exp
        return claims


class SecretsProvider:
    """Least-privilege secrets loader (env / Vault hook) enforcing per-service allow-lists and log redaction."""

    DEFAULT_ALLOWLIST: Dict[str, Set[str]] = {
        "retrieval_service": {"MILVUS_TOKEN"},
        "observability_service": {"LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"},
        "gateway_service": {"ZERO_TRUST_JWT_SECRET"},
        "sql_opsbot_service": set(),
    }

    def __init__(
        self,
        static_secrets: Optional[Dict[str, str]] = None,
        service_allowlist: Optional[Dict[str, Set[str]]] = None,
    ):
        self._static = dict(static_secrets or {})
        self._allowlist = dict(service_allowlist or self.DEFAULT_ALLOWLIST)

    def register_secret(
        self,
        secret_name: str,
        value: str,
        allowed_services: Optional[Set[str]] = None,
    ) -> None:
        """Registers a managed runtime secret and optional per-service allow-list entries."""
        self._static[secret_name] = value
        for svc in allowed_services or set():
            self._allowlist.setdefault(svc, set()).add(secret_name)

    def get_secret(self, service_name: str, secret_name: str) -> str:
        allowed_keys = self._allowlist.get(service_name, set())
        if secret_name not in allowed_keys:
            raise PermissionError(
                f"Least-privilege violation: service '{service_name}' is not authorized to read '{secret_name}'."
            )
        if secret_name in self._static:
            return self._static[secret_name]
        val = os.getenv(secret_name, "")
        if not val:
            raise KeyError(f"Secret '{secret_name}' not configured.")
        return val

    def redact_secrets(self, text: str) -> str:
        """Scrubs JWTs, known API key formats, and registered secret values before emitting logs/traces."""
        redacted = JWT_RE.sub("[REDACTED_SECRET]", text or "")
        redacted = SECRET_PATTERN_RE.sub("[REDACTED_SECRET]", redacted)
        for val in self._static.values():
            if val and len(val) >= 6:
                redacted = redacted.replace(val, "[REDACTED_SECRET]")
        return redacted


zero_trust_auth = ZeroTrustTokenAuthority()
secrets_provider = SecretsProvider()
