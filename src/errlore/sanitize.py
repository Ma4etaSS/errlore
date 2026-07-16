"""Common text sanitizers for lesson and error description content.

Shared logic for :func:`sanitize_lesson_text` (this module) and
:func:`~errlore.errmem.injector.sanitize_description` (which imports
the JSON-extraction helper from here).
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_JSON_FIELD_PATTERNS: tuple[str, ...] = (
    r'"message"\s*:\s*"([^"]{3,})"',
    r'"error"\s*:\s*"([^"]{3,})"',
    r'"description"\s*:\s*"([^"]{3,})"',
)


def extract_readable_from_json(text: str) -> str | None:
    """Try to extract a human-readable field from JSON-like text.

    Searches for ``message``, ``error``, or ``description`` string fields
    and returns the first match with 3+ characters.

    Args:
        text: Raw JSON-like string.

    Returns:
        Extracted readable string, or None if no suitable field found.
    """
    for pattern in _JSON_FIELD_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# sanitize_lesson_text
# ---------------------------------------------------------------------------

# Fenced code blocks: ```lang\n...\n```
_CODE_FENCE_RE: re.Pattern[str] = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)

# Raw JSON-like text (object or array), no backtick — code fences
# are already stripped before this check fires.
_JSON_LIKE_RE: re.Pattern[str] = re.compile(r"^\s*[\{\[]")

# Collapse runs of whitespace into a single space.
_COLLAPSE_WS_RE: re.Pattern[str] = re.compile(r"\s+")

# Non-printable control characters (C0 minus \t\n\r, plus DEL). These survive
# an \s-only collapse and can carry ANSI escape / NUL payloads into the prompt.
_CONTROL_CHARS_RE: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Zero-width and bidi-control characters. These are invisible, survive an
# \s-only collapse, and let an attacker split a keyword ("ig<ZWSP>nore") to
# slip past the override-phrase filter -- so they are removed outright.
_ZERO_WIDTH_RE: re.Pattern[str] = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069\ufeff]"
)

# Prompt-injection override phrases. Lessons are auto-derived from tool output
# (e.g. a failing command's stderr), so their text is only semi-trusted: an
# attacker who can influence a captured failure could plant an instruction that
# later lands in another session's context. This filter redacts the high-signal
# "override the instructions above" family with ``[redacted]``.
#
# It is DEFENSE-IN-DEPTH, not a complete semantic gate: a determined attacker
# can phrase an override the pattern does not list. The load-bearing control is
# the "treat as data, not instructions" framing of the injected block (see
# AgentMemory.inject_for). The patterns below are tuned to redact the obvious
# payloads while NOT mangling legitimate lessons -- so they require an explicit
# "<override-verb> ... <previous/above> ... <instruction-noun>" structure rather
# than any single trigger word. Callers must NFKC-normalize and strip zero-width
# chars first (sanitize_lesson_text does) so homoglyph/full-width/ZWSP variants
# fold to ASCII before matching.
_INJECTION_OVERRIDE_RE: re.Pattern[str] = re.compile(
    r"""
    (?:ignore|disregard|override|forget
        |pay\s+no\s+attention\s+to|stop\s+following|do\s+not\s+follow)\s+
        (?:all\s+|any\s+|the\s+|your\s+)*
        (?:previous|prior|above|preceding|earlier|foregoing)\s+
        (?:\w+\s+){0,2}?
        (?:instructions?|context|prompts?|messages?|rules?|directives?
            |guidance|guardrails?|commands?)
    | </?\s*(?:system|assistant|user|instructions?)\s*>
    """,
    re.IGNORECASE | re.VERBOSE,
)


def neutralize_injection(text: str) -> str:
    """Redact prompt-injection override phrases from *text*.

    Replaces the high-signal "ignore previous instructions" family (and a few
    role-delimiter spoofs) with ``[redacted]``. Defense-in-depth, not a complete
    gate -- see :data:`_INJECTION_OVERRIDE_RE`. For best coverage the caller
    should NFKC-normalize and strip zero-width chars first (as
    :func:`sanitize_lesson_text` does), so this is also safe to call on
    already-normalized text.
    """
    return _INJECTION_OVERRIDE_RE.sub("[redacted]", text)


def _truncate_at_word(text: str, max_len: int) -> str:
    """Truncate *text* at a word boundary, appending ``...`` if needed.

    Guarantees the result is at most *max_len* characters.
    """
    if len(text) <= max_len:
        return text
    suffix = "..."
    cut = max_len - len(suffix)
    idx = text.rfind(" ", 0, cut)
    if idx <= 0:
        # No space found — hard cut.
        return text[:cut] + suffix
    return text[:idx] + suffix


def sanitize_lesson_text(text: str, *, max_len: int = 300) -> str | None:
    """Sanitize lesson text for prompt injection.

    Rules applied in order:

    1. Strip outer whitespace.
    2. Remove fenced code blocks (triple-backtick blocks).
       If the entire text was code, return ``None``.
    3. If the remainder looks like raw JSON (starts with ``{`` or ``[``),
       try to extract a readable field; otherwise return ``None``.
    4. Collapse whitespace.
    5. Truncate at word boundary with ``...``.

    Args:
        text: Raw lesson text.
        max_len: Maximum length of the result (default 300).

    Returns:
        Cleaned text, or ``None`` if the input is raw JSON or code-only.
    """
    # Fold homoglyph / full-width / compatibility variants to ASCII so the
    # override-phrase filter cannot be bypassed with lookalike characters
    # (full-width Latin letters, a Cyrillic "i", etc.). NFKC is idempotent and
    # leaves normal text untouched.
    text = unicodedata.normalize("NFKC", text)
    # Remove zero-width / bidi controls that could split a keyword invisibly.
    text = _ZERO_WIDTH_RE.sub("", text)
    # B8: strip BOM (UTF-8 byte-order mark) if present.
    text = text.lstrip("﻿")
    # Drop non-printable control chars (ANSI escapes, NUL) before anything else.
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.strip()
    if not text:
        return None

    # Strip code fences (keep surrounding prose).
    stripped = _CODE_FENCE_RE.sub("", text).strip()
    if not stripped:
        # Entire text was fenced code blocks.
        return None

    # Raw JSON / array detection.
    if _JSON_LIKE_RE.match(stripped):
        extracted = extract_readable_from_json(stripped)
        if extracted is None:
            return None
        stripped = extracted

    # Collapse whitespace.
    stripped = _COLLAPSE_WS_RE.sub(" ", stripped).strip()
    if not stripped:
        return None

    # Neutralize prompt-injection override phrases before the text can reach a
    # prompt. Applied after collapse so payloads split by odd whitespace
    # ("ignore   previous  instructions") are still caught.
    stripped = neutralize_injection(stripped).strip()
    if not stripped:
        return None

    return _truncate_at_word(stripped, max_len)
