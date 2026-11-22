"""P4.4 Zero Trust Short-Lived Service JWTs & Least-Privilege Secrets Management."""

from app.security.zero_trust import (
    SecretsProvider,
    ZeroTrustError,
    ZeroTrustTokenAuthority,
    secrets_provider,
    zero_trust_auth,
)

__all__ = [
    "SecretsProvider",
    "ZeroTrustError",
    "ZeroTrustTokenAuthority",
    "secrets_provider",
    "zero_trust_auth",
]
