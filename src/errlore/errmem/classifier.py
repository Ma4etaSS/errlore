"""Auto-classification of error types from string messages and stacktraces.

Solves the generic "Error" problem (61 % of records in NEXUS audit) by
extracting a concrete type (ImportError, TimeoutError, etc.) from text.
"""

from __future__ import annotations

import re

# Captures: ValueError, TimeoutError, ConnectionException, RuntimeWarning, etc.
_ERROR_TYPE_RE = re.compile(
    r"\b([A-Z]\w*(?:Error|Exception|Warning|Timeout|Fault|Failure))\b",
)


def classify_error(
    error: Exception | None = None,
    message: str = "",
    stacktrace: str = "",
) -> str:
    """Extract the most specific error type from available information.

    Priority:
        1. Exception class name (if an object is provided).
        2. Last match in the stacktrace (most specific).
        3. First match in the message text.
        4. Fallback: ``"UnclassifiedError"``.

    Args:
        error: Exception object, if available.
        message: Textual error description.
        stacktrace: Full stacktrace string.

    Returns:
        Concrete error type name, e.g. ``"ImportError"``.
    """
    # 1. Direct extraction from exception object
    if error is not None:
        return type(error).__name__

    # 2. Stacktrace — search from the end (last error is the most specific)
    if stacktrace:
        matches = _ERROR_TYPE_RE.findall(stacktrace)
        if matches:
            return str(matches[-1])

    # 3. Message — first occurrence
    if message:
        match = _ERROR_TYPE_RE.search(message)
        if match:
            return match.group(1)

    return "UnclassifiedError"
