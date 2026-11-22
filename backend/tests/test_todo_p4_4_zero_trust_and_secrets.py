"""TDD tests for P4.4: Zero Trust Short-Lived Service JWTs (jti replay protection) & Least-Privilege Secrets."""

import time
from fastapi.testclient import TestClient

from main import app
from app.security.zero_trust import SecretsProvider, ZeroTrustTokenAuthority, ZeroTrustError


client = TestClient(app)


def test_zero_trust_short_lived_jwt_verifies_blocks_tampering_expiry_and_jti_replay():
    """Zero Trust JWT: issues HS256 <=300s tokens, blocks tampered/expired tokens, and stops jti replay attacks."""
    auth = ZeroTrustTokenAuthority(signing_secret="test-zero-trust-secret-key-2026")

    tok = auth.issue_token(
        subject="retrieval-service",
        audience="rag-gateway-api",
        roles=["public", "engineering"],
        ttl_sec=60,
    )
    claims = auth.verify_token(tok, expected_audience="rag-gateway-api", prevent_replay=True)
    assert claims["sub"] == "retrieval-service"
    assert "engineering" in claims["roles"]

    # Replaying the same jti must fail with ZeroTrustError
    replay_blocked = False
    try:
        auth.verify_token(tok, expected_audience="rag-gateway-api", prevent_replay=True)
    except ZeroTrustError as exc:
        replay_blocked = "replay" in str(exc).lower()
    assert replay_blocked is True

    # Tampering with payload breaks HMAC signature
    parts = tok.split(".")
    tampered = f"{parts[0]}.{parts[1]}X.{parts[2]}"
    tamper_blocked = False
    try:
        auth.verify_token(tampered, expected_audience="rag-gateway-api")
    except ZeroTrustError:
        tamper_blocked = True
    assert tamper_blocked is True

    # Expired token rejected
    short_tok = auth.issue_token(subject="sql-opsbot", audience="rag-gateway-api", roles=["public"], ttl_sec=0.02)
    time.sleep(0.05)
    exp_blocked = False
    try:
        auth.verify_token(short_tok, expected_audience="rag-gateway-api")
    except ZeroTrustError as exc:
        exp_blocked = "expired" in str(exc).lower()
    assert exp_blocked is True


def test_least_privilege_secrets_provider_enforces_scope_and_redacts_leaks():
    """SecretsProvider isolates secret scopes per service and redacts leaked tokens/keys from logs."""
    sp = SecretsProvider(
        static_secrets={
            "MILVUS_TOKEN": "milvus_super_secret_991",
            "LANGFUSE_SECRET_KEY": "sk-lf-topsecret-882",
        },
        service_allowlist={
            "retrieval_service": {"MILVUS_TOKEN"},
            "observability_service": {"LANGFUSE_SECRET_KEY"},
        },
    )
    assert sp.get_secret("retrieval_service", "MILVUS_TOKEN") == "milvus_super_secret_991"

    denied = False
    try:
        sp.get_secret("retrieval_service", "LANGFUSE_SECRET_KEY")
    except PermissionError:
        denied = True
    assert denied is True

    raw_log = "Auth failed with Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.sig123 and milvus_super_secret_991"
    redacted = sp.redact_secrets(raw_log)
    assert "milvus_super_secret_991" not in redacted
    assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_security_token_and_verify_http_endpoints():
    """POST /security/token issues a short-lived JWT and POST /security/verify validates it via X-Service-Token."""
    r_tok = client.post(
        "/security/token",
        json={"subject": "peer-ops-agent", "audience": "rag-gateway-api", "roles": ["public", "engineering"], "ttl_sec": 120},
    )
    assert r_tok.status_code == 200
    token = r_tok.json()["token"]
    assert token.count(".") == 2

    r_ver = client.post("/security/verify", headers={"X-Service-Token": token})
    assert r_ver.status_code == 200
    assert r_ver.json()["valid"] is True
    assert r_ver.json()["claims"]["sub"] == "peer-ops-agent"

    # Replaying the exact same single-use token returns 401
    r_replay = client.post("/security/verify", headers={"X-Service-Token": token})
    assert r_replay.status_code == 401
