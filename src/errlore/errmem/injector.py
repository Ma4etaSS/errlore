"""Warning injector — builds prompt warnings from error memory.

Formats known model weaknesses and past errors into a structured
block suitable for system-prompt injection.
"""

from __future__ import annotations

import logging
import re

from errlore.errmem.patterns import PatternDetector
from errlore.errmem.tracker import ErrorTracker
from errlore.sanitize import extract_readable_from_json

logger = logging.getLogger("errlore.errmem")

_MAX_DESCRIPTION_LEN = 200
_RAW_JSON_RE = re.compile(r"^\s*[\{`]")


def sanitize_description(text: str) -> str | None:
    """Sanitize a description for human-readable prompt injection.

    Rules:
        - Strip to ``_MAX_DESCRIPTION_LEN`` characters.
        - Collapse runs of whitespace into a single space.
        - If the text looks like raw JSON (starts with ``{`` or backticks),
          try to extract a readable ``"message"`` / ``"error"`` field;
          otherwise discard the entry entirely (return ``None``).

    Args:
        text: Raw description string.

    Returns:
        Cleaned string, or ``None`` if the input is unsalvageable junk.
    """
    text = text.strip()
    if not text:
        return None

    if _RAW_JSON_RE.match(text):
        extracted = extract_readable_from_json(text)
        if extracted is None:
            return None
        text = extracted

    # B8: collapse whitespace (consistent with sanitize.py).
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > _MAX_DESCRIPTION_LEN:
        text = text[: _MAX_DESCRIPTION_LEN - 3] + "..."

    return text


class WarningInjector:
    """Build prompt-injection warnings from error memory.

    Combines model weaknesses (frequent error types) and past errors
    on similar task types into a ``KNOWN ISSUES`` block.

    auto_tuner_warnings from NEXUS are intentionally **not** ported
    (dead bridge, NEXUS-specific coupling).

    Args:
        tracker: :class:`ErrorTracker` instance.
        detector: :class:`PatternDetector` instance.
        top_n: Maximum number of weakness lines to include.
            Defaults to ``3``.
    """

    def __init__(
        self,
        tracker: ErrorTracker,
        detector: PatternDetector,
        *,
        top_n: int = 3,
    ) -> None:
        self._tracker = tracker
        self._detector = detector
        self._top_n = top_n

    def build_warning(self, model: str, task_type: str) -> str:
        """Build a warning string for prompt injection.

        Output format (verbatim)::

            KNOWN ISSUES:
            - {error_type} (x{count})
            - Past error on similar task: {description}

        Args:
            model: Model name.
            task_type: Task category.

        Returns:
            Warning string, or ``""`` if there is nothing to report.
        """
        # C2: single read of model_accuracy.jsonl instead of two separate
        # calls (get_model_weaknesses + get_errors_for_task_type each did
        # a full file read).  The tracker's _read_all is now cached via
        # JSONLWriter.read_all mtime+size cache, so the second call hits
        # the cache automatically.  No API change needed -- the cache in
        # JSONLWriter handles this transparently.
        weaknesses = self._tracker.get_model_weaknesses(model)
        past_errors = self._tracker.get_errors_for_task_type(task_type)

        if not weaknesses and not past_errors:
            return ""

        parts: list[str] = ["KNOWN ISSUES:"]

        for w in weaknesses[: self._top_n]:
            parts.append(f"- {w}")

        for e in past_errors[:2]:
            raw_desc = str(e.get("description", ""))
            desc = sanitize_description(raw_desc)
            if desc is None:
                continue
            parts.append(f"- Past error on similar task: {desc}")

        # If we only have the header line (all past_errors were junk), skip
        if len(parts) <= 1:
            return ""

        return "\n".join(parts)

    def get_penalty(self, model: str, task_type: str) -> float:
        """Calculate an error-history penalty for model routing.

        Combines weakness count and past-error similarity into a
        single ``[0.0, 1.0]`` score.  Higher means more errors
        on record for this ``(model, task_type)`` pair.

        Args:
            model: Model name.
            task_type: Task category used as a similarity probe.

        Returns:
            Penalty score in ``[0.0, 1.0]``.
        """
        errors = self._tracker.get_model_errors(model)
        if not errors:
            return 0.0

        # Component 1: weakness-based penalty (capped at 0.5)
        weaknesses = self._tracker.get_model_weaknesses(model)
        weakness_penalty = min(0.5, len(weaknesses) * 0.1)

        # Component 2: similarity-based penalty (capped at 0.5)
        similarity = self._detector.similarity_score(task_type, errors)
        similarity_penalty = min(0.5, similarity * 0.5)

        return min(1.0, weakness_penalty + similarity_penalty)
