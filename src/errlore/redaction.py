"""Optional redaction of sensitive data before it is written to memory files.

errlore's error descriptions come from tool output (stderr, command lines),
which routinely contains credentials, emails, and addresses. With
``privacy_mode`` enabled on :class:`~errlore.facade.AgentMemory`, every text
field is passed through :class:`Redactor` before it is persisted to
``errors.jsonl`` / ``lessons.jsonl`` -- so secrets never reach disk, and
therefore can never be re-injected into a later prompt.

Default patterns favor precision over recall: they match formats that are
almost certainly sensitive (emails, IPs, bearer headers, well-known key
prefixes, ``password=...`` pairs) and deliberately do NOT match generic long
strings, which in error text are usually hashes or IDs a lesson needs.
Callers extend coverage with their own regexes via ``redact_patterns``.
"""

from __future__ import annotations

import re

# (pattern, replacement) pairs applied in order. Replacements are typed so a
# redacted lesson stays readable ("connect as [REDACTED_EMAIL]" still teaches).
_DEFAULT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Email addresses.
    (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "[REDACTED_EMAIL]",
    ),
    # key=value / key: value credential pairs. Must run before the bare-token
    # patterns so the key name is preserved ("password=[REDACTED]").
    (
        re.compile(
            r"(?i)\b(password|passwd|secret|token|api[_-]?key|access[_-]?key"
            r"|auth|credentials?)\s*[=:]\s*(\"[^\"]+\"|'[^']+'|\S+)"
        ),
        r"\1=[REDACTED]",
    ),
    # Authorization: Bearer <token>.
    (
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
        "[REDACTED_TOKEN]",
    ),
    # Well-known API key prefixes: OpenAI/Anthropic-style, GitHub, AWS, Slack.
    (re.compile(r"\bsk-(?:[A-Za-z0-9_-]+-)?[A-Za-z0-9_-]{16,}\b"), "[REDACTED_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "[REDACTED_KEY]"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED_KEY]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_KEY]"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED_KEY]"),
    # IPv4 addresses. Last: an IP inside a redacted credential is already gone.
    (
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[REDACTED_IP]",
    ),
)


class Redactor:
    """Apply the default + user-supplied redaction patterns to text.

    Args:
        extra_patterns: Optional user regexes (strings). Each match is
            replaced with ``[REDACTED]``. Invalid regexes raise ``re.error``
            at construction -- better to fail at setup than to silently not
            redact at write time.
    """

    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        self._extra: tuple[re.Pattern[str], ...] = tuple(
            re.compile(p) for p in (extra_patterns or [])
        )

    def redact(self, text: str) -> str:
        """Return *text* with all sensitive matches replaced."""
        for pattern, replacement in _DEFAULT_PATTERNS:
            text = pattern.sub(replacement, text)
        for pattern in self._extra:
            text = pattern.sub("[REDACTED]", text)
        return text
