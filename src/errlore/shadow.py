"""Shadow-mode counterfactual queue.

The interference-guard (``errlore.lessons.graduation``) learns from the *live*
loop; shadow mode learns from a **counterfactual** loop that never touches the
user-facing output. The flow (SHADOW_MODE_SPEC.md):

1. Build an injected prompt block with ``inject_for(..., mode="shadow")`` and
   keep it OUT of the main request -- the user's output is untouched.
2. Enqueue the (baseline_prompt, injected_prompt) pair here.
3. Your worker re-runs both against the same model and scores each with the
   surface's **deterministic validator** (schema, sentinel, exit code, tests --
   never an LLM judge).
4. Report the two pass/fail outcomes; per-lesson Beta posteriors update and the
   lesson graduates (promote) / holds / quarantines.

errlore never calls the model or the validator -- that is the worker's job.
This module owns only the durable queue (``counterfactuals.jsonl``), so trials
survive restarts and outcomes can be reported asynchronously.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from errlore.io import JSONLWriter
from errlore.lessons.models import _short_id, _utc_now_iso

logger = logging.getLogger("errlore.shadow")


@dataclass(slots=True)
class Counterfactual:
    """A queued counterfactual trial awaiting a worker result."""

    cf_id: str
    lesson_ids: list[str]
    model: str
    baseline_prompt: str
    injected_prompt: str
    created_at: str


class CounterfactualQueue:
    """Durable JSONL queue of counterfactual trials.

    Append-only log with two event kinds (``queued`` / ``resolved``) addressed
    by ``cf_id``; a trial is pending until a matching ``resolved`` event lands.
    """

    def __init__(self, data_dir: Path | str) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._data_dir / "counterfactuals.jsonl"
        self._writer = JSONLWriter(max_bytes=None)
        self._lock = threading.Lock()

    def enqueue(
        self,
        lesson_ids: list[str],
        model: str,
        baseline_prompt: str,
        injected_prompt: str,
    ) -> str:
        """Queue a trial. Returns its ``cf_id``."""
        cf_id = _short_id()
        self._writer.append(
            self._path,
            {
                "event": "queued",
                "cf_id": cf_id,
                "lesson_ids": list(lesson_ids),
                "model": model,
                "baseline_prompt": baseline_prompt,
                "injected_prompt": injected_prompt,
                "created_at": _utc_now_iso(),
            },
        )
        return cf_id

    def pending(self) -> list[Counterfactual]:
        """Return trials queued but not yet resolved."""
        events = self._writer.read_all(self._path)
        resolved: set[str] = set()
        queued: dict[str, dict[str, object]] = {}
        for ev in events:
            cf_id = str(ev.get("cf_id", ""))
            if ev.get("event") == "resolved":
                resolved.add(cf_id)
            elif ev.get("event") == "queued":
                queued[cf_id] = ev

        result: list[Counterfactual] = []
        for cf_id, ev in queued.items():
            if cf_id in resolved:
                continue
            raw_ids = ev.get("lesson_ids", [])
            lesson_ids = (
                [str(x) for x in raw_ids] if isinstance(raw_ids, list) else []
            )
            result.append(
                Counterfactual(
                    cf_id=cf_id,
                    lesson_ids=lesson_ids,
                    model=str(ev.get("model", "")),
                    baseline_prompt=str(ev.get("baseline_prompt", "")),
                    injected_prompt=str(ev.get("injected_prompt", "")),
                    created_at=str(ev.get("created_at", "")),
                )
            )
        return result

    def resolve(
        self,
        cf_id: str,
        baseline_passed: bool,
        injected_passed: bool,
    ) -> list[str] | None:
        """Resolve a queued trial.

        Idempotent under a cross-process file lock (mirrors
        ``report_outcome``): the whole find -> dup-check -> append runs while
        the lock is held.

        Returns:
            The trial's ``lesson_ids`` on the first resolve, or ``None`` if the
            trial was already resolved (duplicate).

        Raises:
            KeyError: If *cf_id* was never queued.
        """
        with self._lock, self._writer.lock(self._path):
            # Read fresh under the lock -- a stale cross-process cache could miss
            # another worker's "resolved" marker and double-resolve, corrupting
            # the graduation posterior (same fix as AgentMemory.report_outcome).
            events = self._writer.read_all(self._path, use_cache=False)

            # Duplicate check FIRST: after compaction the (redundant) queued
            # record of a closed trial is gone, and a re-resolve must still
            # return None rather than raise KeyError.
            for ev in events:
                if ev.get("event") == "resolved" and ev.get("cf_id") == cf_id:
                    logger.warning("Duplicate resolve for counterfactual %s", cf_id)
                    return None

            queued: dict[str, object] | None = None
            for ev in events:
                if ev.get("event") == "queued" and ev.get("cf_id") == cf_id:
                    queued = ev
                    break
            if queued is None:
                raise KeyError(f"Unknown counterfactual: {cf_id}")

            self._writer.append(
                self._path,
                {
                    "event": "resolved",
                    "cf_id": cf_id,
                    "baseline_passed": baseline_passed,
                    "injected_passed": injected_passed,
                    "resolved_at": _utc_now_iso(),
                },
            )

            # Bound growth: queued records carry two full prompts each, so a
            # long-lived queue is dominated by closed trials' dead weight.
            # Still holding the file lock here.
            if len(events) + 1 >= self._COMPACT_THRESHOLD:
                self._compact()

            raw_ids = queued.get("lesson_ids", [])
            return [str(x) for x in raw_ids] if isinstance(raw_ids, list) else []

    # Compact once the log exceeds this many records (mirrors AgentMemory's
    # injections compaction).
    _COMPACT_THRESHOLD: int = 1000

    def _compact(self) -> None:
        """Drop the queued record of every resolved trial (keep its marker).

        A closed trial only needs its small ``resolved`` marker for the
        duplicate-resolve check; the ``queued`` record -- which carries both
        full prompts -- is dead weight. Pending trials are kept in full.
        Must be called while holding the counterfactuals file lock.
        """
        def transform(
            events: list[dict[str, object]],
        ) -> list[dict[str, object]] | None:
            resolved: set[str] = {
                str(ev.get("cf_id", ""))
                for ev in events
                if ev.get("event") == "resolved"
            }
            kept = [
                ev
                for ev in events
                if not (
                    ev.get("event") == "queued"
                    and str(ev.get("cf_id", "")) in resolved
                )
            ]
            return kept if len(kept) != len(events) else None

        try:
            self._writer.atomic_update(self._path, transform)
        except Exception:  # pragma: no cover - compaction must never break loop
            logger.warning("counterfactuals compaction failed; continuing", exc_info=True)
