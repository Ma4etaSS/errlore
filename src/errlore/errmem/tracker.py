"""Model error tracker (Amygdala) — records errors and builds weakness profiles.

Persists data to ``model_accuracy.jsonl`` via :class:`errlore.io.JSONLWriter`.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from errlore.io import JSONLWriter

logger = logging.getLogger("errlore.errmem")


class ErrorTracker:
    """Track per-model errors and build weakness profiles.

    All data is stored in a single append-only JSONL file
    (``data_dir / "model_accuracy.jsonl"``).

    Args:
        data_dir: Directory for persistent storage.
        min_occurrences: Minimum repeat count to consider something a weakness.
            Defaults to ``3``.
        writer: Optional :class:`JSONLWriter` instance (shared across modules).
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        min_occurrences: int = 3,
        writer: JSONLWriter | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._accuracy_file = self._data_dir / "model_accuracy.jsonl"
        self._lock = threading.Lock()
        self._min_occurrences = min_occurrences
        self._writer = writer or JSONLWriter()

    # -- write ----------------------------------------------------------

    def record_error(
        self,
        model: str,
        task_type: str,
        error: dict[str, Any],
    ) -> dict[str, Any]:
        """Record a model error.

        Args:
            model: Model name.
            task_type: Task category.
            error: Dict with ``type``, ``description``, ``severity`` keys.

        Returns:
            The persisted entry dict.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "task_type": task_type,
            "error_type": error.get("type", "unknown"),
            "description": error.get("description", ""),
            "severity": error.get("severity", "medium"),
        }
        self._writer.append(self._accuracy_file, entry)
        return entry

    # -- read -----------------------------------------------------------

    def get_model_errors(self, model: str) -> list[dict[str, Any]]:
        """Return all recorded errors for a specific model."""
        all_entries = self._read_all()
        return [e for e in all_entries if e.get("model") == model]

    def get_errors_for_task_type(self, task_type: str) -> list[dict[str, Any]]:
        """Return errors for a specific task type."""
        all_entries = self._read_all()
        return [e for e in all_entries if e.get("task_type") == task_type]

    def get_model_weaknesses(
        self,
        model: str,
        min_occurrences: int | None = None,
    ) -> list[str]:
        """Return known weaknesses for a model based on error patterns.

        Args:
            model: Model name.
            min_occurrences: Override the instance default threshold.

        Returns:
            List of weakness descriptions like ``"TimeoutError (x5)"``.
        """
        if min_occurrences is None:
            min_occurrences = self._min_occurrences

        errors = self.get_model_errors(model)
        if not errors:
            return []

        counter: Counter[str] = Counter()
        for e in errors:
            error_type = e.get("error_type", "unknown")
            counter[error_type] += 1

        weaknesses: list[str] = []
        for error_type, count in counter.most_common():
            if count >= min_occurrences:
                weaknesses.append(f"{error_type} (x{count})")

        return weaknesses

    def get_model_profile(self, model: str) -> dict[str, Any]:
        """Return an aggregate error profile for a model.

        Returns:
            Dict with ``errors`` (total count), ``last_error`` (ISO timestamp),
            and ``weaknesses`` (list of weakness strings).
        """
        errors = self.get_model_errors(model)
        profile: dict[str, Any] = {
            "errors": len(errors),
            "last_error": errors[-1].get("timestamp", "") if errors else "",
            "weaknesses": self.get_model_weaknesses(model),
        }
        return profile

    def get_model_stats(self) -> dict[str, dict[str, Any]]:
        """Return per-model error statistics.

        Returns:
            ``{model_name: {"errors": N, "last_error": "timestamp"}}``.
        """
        stats: dict[str, dict[str, Any]] = {}
        all_entries = self._read_all()
        for entry in all_entries:
            model = str(entry.get("model", "unknown"))
            if model not in stats:
                stats[model] = {"errors": 0, "last_error": ""}
            stats[model]["errors"] += 1
            stats[model]["last_error"] = str(entry.get("timestamp", ""))
        return stats

    # -- internals ------------------------------------------------------

    def _read_all(self) -> list[dict[str, Any]]:
        """Read all entries from the accuracy file (thread-safe)."""
        with self._lock:
            return self._writer.read_all(self._accuracy_file)
