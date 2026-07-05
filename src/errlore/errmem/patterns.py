"""Error pattern detection with RU/EN stemming-based similarity.

Groups errors by ``(model, error_type, task_type)`` and surfaces patterns
whose frequency meets a configurable threshold.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_CYRILLIC_RE = re.compile(r"^[а-яёА-ЯЁ]+$")

_STOP_WORDS_RU = frozenset(
    {
        "и",
        "в",
        "на",
        "с",
        "по",
        "для",
        "из",
        "к",
        "от",
        "до",
        "не",
        "но",
        "а",
        "что",
        "как",
        "это",
        "при",
        "за",
        "или",
        "же",
        "ли",
        "бы",
        "то",
        "его",
        "её",
        "их",
        "он",
        "она",
        "мы",
        "вы",
        "они",
        "был",
        "быть",
        "будет",
        "если",
        "так",
        "уже",
        "ещё",
        "тоже",
        "также",
        "только",
        "нет",
        "да",
    },
)

_STOP_WORDS_EN = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "or",
        "not",
        "but",
        "if",
        "it",
        "this",
        "that",
    },
)


def _stem_ru(word: str) -> str:
    """Naive suffix stemmer for Russian words."""
    if len(word) <= 4 or not _CYRILLIC_RE.match(word):
        return word
    for suffix_len in (3, 2):
        if len(word) > suffix_len + 2:
            return word[:-suffix_len]
    return word


def _normalize_words(text: str) -> set[str]:
    """Lowercase, stem, and remove stop-words for RU/EN text."""
    words = set(text.lower().split())
    words -= _STOP_WORDS_RU
    words -= _STOP_WORDS_EN
    return {_stem_ru(w) for w in words if len(w) > 1}


class PatternDetector:
    """Detect recurring error patterns from a list of error entries.

    Groups by ``(model, error_type, task_type)`` and returns groups
    whose frequency is ``>= min_occurrences``.

    Args:
        min_occurrences: Minimum repeat count to be considered a pattern.
            Defaults to ``3``.
    """

    def __init__(self, min_occurrences: int = 3) -> None:
        self._min_occurrences = min_occurrences

    def detect(self, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Detect recurring error patterns.

        Args:
            errors: List of error entry dicts.

        Returns:
            List of pattern dicts with ``model``, ``error_type``,
            ``task_type``, and ``occurrences`` keys.
        """
        counter: Counter[tuple[str, str, str]] = Counter()
        for e in errors:
            key = (
                str(e.get("model", "unknown")),
                str(e.get("error_type", "unknown")),
                str(e.get("task_type", "unknown")),
            )
            counter[key] += 1

        patterns: list[dict[str, Any]] = []
        for (model, error_type, task_type), count in counter.most_common():
            if count >= self._min_occurrences:
                patterns.append(
                    {
                        "model": model,
                        "error_type": error_type,
                        "task_type": task_type,
                        "occurrences": count,
                    },
                )

        return patterns

    def similarity_score(
        self,
        task_description: str,
        past_errors: list[dict[str, Any]],
    ) -> float:
        """Estimate similarity of a new task to past errors.

        Uses word-overlap heuristic with RU/EN stemming and stop-word removal.

        Args:
            task_description: Description of the current task.
            past_errors: List of past error entries.

        Returns:
            Similarity score in ``[0.0, 1.0]``.
        """
        if not past_errors or not task_description:
            return 0.0

        task_words = _normalize_words(task_description)
        if not task_words:
            return 0.0

        max_overlap = 0.0
        for error in past_errors:
            description = str(error.get("description", ""))
            task_type = str(error.get("task_type", ""))
            error_words = _normalize_words(f"{description} {task_type}")
            if not error_words:
                continue
            overlap = len(task_words & error_words) / len(task_words)
            max_overlap = max(max_overlap, overlap)

        return min(1.0, max_overlap)
