"""Unified AgentMemory facade -- single entry point for error learning.

Composes :class:`~errlore.lessons.store.LessonStore`,
:class:`~errlore.errmem.tracker.ErrorTracker`,
:class:`~errlore.errmem.patterns.PatternDetector`,
:class:`~errlore.errmem.injector.WarningInjector`, and
:class:`~errlore.trust.engine.TrustEngine` into one coherent API with
a closed reinforcement loop.

Injection handles are persisted to ``data_dir/injections.jsonl``
(append-only), so outcomes can be reported even after process restart.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from errlore.errmem import ErrorTracker, PatternDetector, WarningInjector, classify_error
from errlore.io import JSONLWriter
from errlore.lessons.models import _short_id, _utc_now_iso
from errlore.lessons.store import LessonStore
from errlore.sanitize import sanitize_lesson_text
from errlore.trust import FeedbackSignal, TrustEngine

logger = logging.getLogger("errlore.facade")


# ---------------------------------------------------------------------------
# Injection dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Injection:
    """Result of :meth:`AgentMemory.inject_for`.

    Carries the prompt-ready text block together with enough metadata
    for the reinforcement loop (:meth:`AgentMemory.report_outcome`).

    Attributes:
        handle_id: Unique 12-char hex ID persisted in injections.jsonl.
        text: Assembled prompt block (lessons + known issues).
        lesson_ids: IDs of lessons included in this injection.
        model: Model the injection was built for.
        domain: Trust domain.
        created_at: ISO-8601 timestamp.
    """

    handle_id: str
    text: str
    lesson_ids: list[str]
    model: str
    domain: str
    created_at: str


# ---------------------------------------------------------------------------
# AgentMemory facade
# ---------------------------------------------------------------------------


class AgentMemory:
    """Unified facade for error memory, lessons, and trust.

    Composes all errlore subsystems behind four core methods:

    * :meth:`log_error` -- record an error.
    * :meth:`resolve` -- mark error as fixed, optionally extract a lesson.
    * :meth:`inject_for` -- build a context block for the next task.
    * :meth:`report_outcome` -- close the reinforcement loop.

    Args:
        data_dir: Directory for all persistent state files.
        trust: Enable the TrustEngine layer (default True).
            When False, :meth:`report_outcome` skips trust updates and
            :meth:`stats` omits the ``trust`` key.
        max_lessons: Maximum lessons to include in injection text.
        decay_every: Run :meth:`LessonStore.decay_unused` every N
            :meth:`inject_for` calls.  Counter is in-process only --
            no cross-restart persistence needed.
        embeddings: Enable semantic retrieval via embeddings (default False).
            Requires ``errlore[embeddings]`` extras (fastembed + numpy).
            When the extras are missing, falls back to word-overlap with
            a warning.
    """

    def __init__(
        self,
        data_dir: Path | str,
        *,
        trust: bool = True,
        max_lessons: int = 3,
        decay_every: int = 50,
        embeddings: bool = False,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._max_lessons = max_lessons
        self._decay_every = decay_every
        self._decay_counter = 0

        # Shared writer for injections.jsonl.
        self._writer = JSONLWriter()
        self._injections_path = self._data_dir / "injections.jsonl"

        # Subsystems.
        retriever = self._build_retriever(self._data_dir) if embeddings else None
        self._store = LessonStore(self._data_dir, retriever=retriever)
        self._tracker = ErrorTracker(self._data_dir)
        self._detector = PatternDetector()
        self._injector = WarningInjector(
            self._tracker, self._detector, top_n=max_lessons,
        )
        self._trust: TrustEngine | None = None
        if trust:
            self._trust = TrustEngine(
                state_path=self._data_dir / "trust.json",
            )

        # Lock for report_outcome idempotency (within one process).
        self._report_lock = threading.Lock()

    @staticmethod
    def _build_retriever(data_dir: Path) -> Any:
        """Try to build a FastEmbedBackend + VectorIndex retriever.

        Returns the VectorIndex or None (with a warning) if the extras
        are not installed.
        """
        try:
            from errlore.retrieval.backend import FastEmbedBackend
            from errlore.retrieval.index import VectorIndex

            backend = FastEmbedBackend()
            return VectorIndex(data_dir, backend)
        except ImportError:
            import warnings

            warnings.warn(
                "errlore[embeddings] extras not installed; "
                "falling back to word-overlap retrieval.  "
                "Install with:  pip install errlore[embeddings]",
                UserWarning,
                stacklevel=3,
            )
            return None

    # ------------------------------------------------------------------
    # Error lifecycle
    # ------------------------------------------------------------------

    def log_error(
        self,
        model: str,
        task_type: str,
        error: BaseException | str,
        *,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record an error and update the model's weakness profile.

        The error is auto-classified (see :func:`errlore.errmem.classify_error`)
        and written to both :class:`LessonStore` (``errors.jsonl``) and
        :class:`ErrorTracker` (``model_accuracy.jsonl``).

        Args:
            model: Model name.
            task_type: Task category (e.g. ``"extraction"``).
            error: Exception object or textual error description.
            message: Optional human-readable override for the stored message.
            metadata: Optional extra data to attach to the error record.

        Returns:
            Error ID (12-char hex).
        """
        if isinstance(error, BaseException):
            error_type = type(error).__name__
            error_message = message or str(error)
        else:
            error_type = classify_error(message=error)
            error_message = message or error

        # LessonStore: full error record.
        err_id = self._store.log_error(
            model=model,
            task_type=task_type,
            error_type=error_type,
            message=error_message,
            metadata=metadata,
        )

        # ErrorTracker: weakness profile.
        self._tracker.record_error(
            model,
            task_type,
            {"type": error_type, "description": error_message, "severity": "medium"},
        )

        return err_id

    def resolve(
        self,
        err_id: str,
        resolution: str,
        lesson: str | None = None,
    ) -> bool:
        """Resolve an error, optionally extracting a lesson.

        When *lesson* is provided it is sanitized via
        :func:`~errlore.sanitize.sanitize_lesson_text`.  If sanitization
        rejects the text (raw JSON, code-only), the lesson is dropped
        with a warning and the error is still resolved.

        Args:
            err_id: Error ID returned by :meth:`log_error`.
            resolution: Free-text resolution description.
            lesson: Optional lesson text to persist.

        Returns:
            True if the error was found (and resolved), False if unknown ID.
        """
        if lesson is not None:
            sanitized = sanitize_lesson_text(lesson)
            if sanitized is None:
                logger.warning(
                    "Lesson text rejected by sanitizer for error %s", err_id,
                )
                lesson = None
            else:
                lesson = sanitized

        return self._store.resolve_error(err_id, resolution, lesson)

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def inject_for(
        self,
        task: str,
        model: str,
        *,
        task_type: str | None = None,
        domain: str | None = None,
    ) -> Injection:
        """Build a context injection block for a new task.

        Searches past lessons and known model weaknesses, assembles them
        into a prompt-ready text block, and persists the injection handle
        so outcomes can be reported later (even after restart).

        **Lazy decay**: every *decay_every* calls, runs
        :meth:`LessonStore.decay_unused` to lower confidence of
        never-applied lessons.  Counter is in-process only.

        Args:
            task: Task description (used as search query for lessons).
            model: Model name (used for known-issue lookup).
            task_type: Optional task category for narrower lesson search.
            domain: Trust domain (default ``"general"``).

        Returns:
            :class:`Injection` with text block and handle for
            :meth:`report_outcome`.
        """
        # Lazy decay.
        self._decay_counter += 1
        if self._decay_counter >= self._decay_every:
            self._decay_counter = 0
            self._store.decay_unused()

        effective_domain = domain or "general"
        effective_task_type = task_type or ""

        # Search lessons.
        lessons = self._store.search_lessons(
            query=task,
            task_type=effective_task_type,
            limit=self._max_lessons,
        )
        lesson_ids = [le.id for le in lessons]

        # Build text block.
        parts: list[str] = []

        if lessons:
            parts.append("[LESSONS FROM PAST FAILURES]")
            for le in lessons:
                parts.append(f"- {le.pattern} -> {le.solution}")

        # Known issues (model weaknesses + past errors).
        warning = self._injector.build_warning(model, effective_task_type)
        if warning:
            if parts:
                parts.append("")  # blank separator line
            parts.append(warning)

        text = "\n".join(parts) if parts else ""

        # Create and persist injection.
        handle_id = _short_id()
        created_at = _utc_now_iso()

        inj = Injection(
            handle_id=handle_id,
            text=text,
            lesson_ids=lesson_ids,
            model=model,
            domain=effective_domain,
            created_at=created_at,
        )

        self._writer.append(
            self._injections_path,
            {
                "event": "issued",
                "handle_id": handle_id,
                "lesson_ids": lesson_ids,
                "model": model,
                "domain": effective_domain,
                "created_at": created_at,
                "text": text,
            },
        )

        return inj

    # ------------------------------------------------------------------
    # Reinforcement loop
    # ------------------------------------------------------------------

    def report_outcome(
        self,
        inj_or_handle_id: Injection | str,
        success: bool,
        *,
        outcome: float | None = None,
    ) -> bool:
        """Close the reinforcement loop for a previously injected context.

        For every lesson that was part of the injection, calls
        :meth:`LessonStore.reinforce` (adjusting confidence and
        applied_count).  If the trust layer is enabled, updates the
        model's trust weight via :class:`TrustEngine`.

        **Idempotent**: reporting the same handle twice returns ``False``
        with a warning; no double-reinforcement occurs.

        Args:
            inj_or_handle_id: :class:`Injection` object or its
                ``handle_id`` string.
            success: Whether the task succeeded.
            outcome: Optional quality score in ``[0, 1]`` for the trust
                signal.  Defaults to ``1.0`` when *success* is True,
                ``0.0`` otherwise.

        Returns:
            ``True`` on first report, ``False`` on duplicate.

        Raises:
            KeyError: If the handle_id is unknown.
        """
        if isinstance(inj_or_handle_id, Injection):
            handle_id = inj_or_handle_id.handle_id
        else:
            handle_id = inj_or_handle_id

        with self._report_lock:
            events = self._writer.read_all(self._injections_path)

            # Find the issued event.
            issued: dict[str, object] | None = None
            for ev in events:
                if (
                    ev.get("event") == "issued"
                    and ev.get("handle_id") == handle_id
                ):
                    issued = ev
                    break

            if issued is None:
                raise KeyError(f"Unknown injection handle: {handle_id}")

            # Idempotency check.
            for ev in events:
                if (
                    ev.get("event") == "reported"
                    and ev.get("handle_id") == handle_id
                ):
                    logger.warning(
                        "Duplicate report_outcome for handle %s", handle_id,
                    )
                    return False

            # Reinforce each lesson.
            raw_ids = issued.get("lesson_ids", [])
            lesson_ids: list[str] = (
                [str(x) for x in raw_ids]
                if isinstance(raw_ids, list)
                else []
            )
            for lid in lesson_ids:
                self._store.reinforce(lid, success)

            # Trust update.
            if self._trust is not None:
                model = str(issued.get("model", ""))
                sig_domain = str(issued.get("domain", "general"))
                outcome_val = (
                    outcome
                    if outcome is not None
                    else (1.0 if success else 0.0)
                )
                signal = FeedbackSignal(outcome=outcome_val, domain=sig_domain)
                self._trust.update(model, signal)
                self._trust.save()

            # Persist reported event.
            self._writer.append(
                self._injections_path,
                {
                    "event": "reported",
                    "handle_id": handle_id,
                    "success": success,
                    "outcome": outcome,
                    "reported_at": _utc_now_iso(),
                },
            )

        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def pending_injections(self) -> list[Injection]:
        """Return all injections issued but not yet reported."""
        events = self._writer.read_all(self._injections_path)

        reported_handles: set[str] = set()
        issued_map: dict[str, dict[str, object]] = {}

        for ev in events:
            event_type = ev.get("event")
            hid = str(ev.get("handle_id", ""))
            if event_type == "reported":
                reported_handles.add(hid)
            elif event_type == "issued":
                issued_map[hid] = ev

        result: list[Injection] = []
        for hid, ev in issued_map.items():
            if hid not in reported_handles:
                raw_ids = ev.get("lesson_ids", [])
                lesson_ids = (
                    [str(x) for x in raw_ids]
                    if isinstance(raw_ids, list)
                    else []
                )
                result.append(
                    Injection(
                        handle_id=hid,
                        text=str(ev.get("text", "")),
                        lesson_ids=lesson_ids,
                        model=str(ev.get("model", "")),
                        domain=str(ev.get("domain", "general")),
                        created_at=str(ev.get("created_at", "")),
                    ),
                )

        return result

    def stats(self) -> dict[str, Any]:
        """Return aggregate statistics.

        Keys: ``errors_total``, ``errors_resolved``, ``errors_unresolved``,
        ``lessons_total``, ``lessons_applied``, ``pending_injections``,
        and (when trust is enabled) ``trust`` — a dict of model weights.
        """
        result: dict[str, Any] = {
            **self._store.counts(),
            "pending_injections": len(self.pending_injections()),
        }
        if self._trust is not None:
            result["trust"] = self._trust.get_weights()
        return result

    def model_penalty(self, model: str, task_type: str) -> float:
        """Return the error-history penalty for a model on a task type.

        Delegates to :meth:`WarningInjector.get_penalty`.

        Args:
            model: Model name.
            task_type: Task category.

        Returns:
            Penalty score in ``[0.0, 1.0]``.
        """
        return self._injector.get_penalty(model, task_type)

    @property
    def trust(self) -> TrustEngine | None:
        """Access the underlying TrustEngine (None when trust=False)."""
        return self._trust

    def best_model(self, domain: str = "general") -> str | None:
        """Return the model with the highest trust weight for a domain.

        Useful for routing: pick the model that historically performs best
        on a given task domain, based on accumulated outcome signals.

        Args:
            domain: Trust domain (default ``"general"``).

        Returns:
            Model name with the highest weight, or None if no models are
            registered or trust is disabled.
        """
        if self._trust is None:
            return None
        weights = self._trust.get_weights(domain)
        if not weights:
            return None
        return max(weights, key=weights.get)  # type: ignore[arg-type]
