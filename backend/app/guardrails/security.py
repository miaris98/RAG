"""Security Guardrails for Enterprise RAG:
1. PII & Secret Redaction (Emails, SSNs, API Keys/Tokens) before indexing or external web search.
2. Indirect Prompt Injection Defense (neutralizes adversarial instructions hidden inside documents).
"""

import re
from typing import List, Tuple


class SecurityGuardrails:
    """Stateless regex & heuristic guardrail sanitizer for ingestion and query paths."""

    EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    API_KEY_PATTERN = re.compile(r"\b(?:sk|pk|api|tok|ghp)_[A-Za-z0-9_-]{12,}\b|\bsk-[A-Za-z0-9]{16,}\b")

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions[^\n]*", re.IGNORECASE),
        re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|system)\s+prompts?[^\n]*", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(?:in\s+)?developer\s+mode[^\n]*", re.IGNORECASE),
        re.compile(r"reveal\s+(?:the\s+)?(?:system\s+prompt|admin\s+passwords?)[^\n]*", re.IGNORECASE),
    ]

    def sanitize_text(
        self,
        text: str,
        redact_pii: bool = True,
        block_injections: bool = True,
    ) -> Tuple[str, List[str]]:
        """Sanitizes text and returns (sanitized_text, triggered_flags)."""
        if not text:
            return text, []

        sanitized = text
        flags: List[str] = []

        if redact_pii:
            pii_found = False
            if self.API_KEY_PATTERN.search(sanitized):
                sanitized = self.API_KEY_PATTERN.sub("[REDACTED_API_KEY]", sanitized)
                pii_found = True
            if self.SSN_PATTERN.search(sanitized):
                sanitized = self.SSN_PATTERN.sub("[REDACTED_SSN]", sanitized)
                pii_found = True
            if self.EMAIL_PATTERN.search(sanitized):
                sanitized = self.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)
                pii_found = True
            if pii_found:
                flags.append("pii_redacted")

        if block_injections:
            injection_found = False
            for pattern in self.INJECTION_PATTERNS:
                if pattern.search(sanitized):
                    sanitized = pattern.sub("[BLOCKED_INJECTION_ATTEMPT]", sanitized)
                    injection_found = True
            if injection_found:
                flags.append("prompt_injection_detected")

        return sanitized, flags


guardrails = SecurityGuardrails()
