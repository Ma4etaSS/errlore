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
from errlore.shadow import CounterfactualQueue
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
        harm_gate: When True (default), lessons whose live failure history
            clears the Beta-Binomial harm bar are withheld from injection
            (see :mod:`errlore.lessons.graduation`).  This targets the
            measured 12-15% interference from injecting lessons into tasks
            they hurt.  A fresh or consistently-helpful lesson is never
            gated, so good lessons are not starved.  Set False to restore
            unconditional injection.
    """

    def __init__(
        self,
        data_dir: Path | str,
        *,
        trust: bool = True,
        max_lessons: int = 3,
        decay_every: int = 50,
        embeddings: bool = False,
        harm_gate: bool = True,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._max_lessons = max_lessons
        self._decay_every = decay_every
        self._decay_counter = 0
        self._harm_gate = harm_gate

        # Shared writer for injections.jsonl -- rotation disabled because
        # injections are addressed by handle_id and must remain visible.
        self._writer = JSONLWriter(max_bytes=None)
        self._injections_path = self._data_dir / "injections.jsonl"

        # Subsystems.
        retriever = self._build_retriever(self._data_dir) if embeddings else None
        self._store = LessonStore(self._data_dir, retriever=retriever)
        self._cf_queue = CounterfactualQueue(self._data_dir)
        self._tracker = ErrorTracker(self._data_dir)
        self._detector = PatternDetector()
        self._injector = WarningInjector(
            self._tracker, self._detector, top_n=max_lessons,
        )
        self._trust: TrustEngine | None = None
        if trust:
            trust_path = self._data_dir / "trust.json"
            # TrustEngine() does NOT read an existing state file -- only the
            # load() classmethod does.  Without this branch, trust weights
            # silently reset on every process restart.
            if trust_path.exists():
                self._trust = TrustEngine.load(trust_path)
            else:
                self._trust = TrustEngine(state_path=trust_path)

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
        mode: str = "live",
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
            mode: ``"live"`` (default) applies the harm gate -- quarantined
                lessons are withheld.  ``"shadow"`` includes ALL candidate
                lessons, quarantined ones too, so a counterfactual trial can
                re-evaluate them off the user-facing path (the recovery route
                for a suppressed lesson).  The returned block should be sent to
                the model only in a shadow run, never the main request.

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

        # Search lessons.  Shadow trials evaluate every candidate (including
        # quarantined ones); only the live path applies the harm gate.
        exclude_quarantined = self._harm_gate and mode != "shadow"
        lessons = self._store.search_lessons(
            query=task,
            task_type=effective_task_type,
            limit=self._max_lessons,
            exclude_quarantined=exclude_quarantined,
        )

        # Build text block.  Sanitize BOTH pattern and solution at the
        # injection boundary -- this is the definitive gate, so a lesson can
        # never carry raw JSON/code/control-chars into the prompt regardless of
        # how it was written (add_lesson's solution, a direct store write, or
        # legacy data all pass through here).  A lesson whose pattern or
        # solution does not survive sanitization is dropped from this
        # injection, and its id is not reinforced.
        parts: list[str] = []
        lesson_ids: list[str] = []
        lesson_lines: list[str] = []
        for le in lessons:
            safe_pattern = sanitize_lesson_text(le.pattern)
            safe_solution = sanitize_lesson_text(le.solution)
            if safe_pattern is None or safe_solution is None:
                logger.warning(
                    "Lesson %s dropped from injection: unsanitizable content", le.id
                )
                continue
            lesson_lines.append(f"- {safe_pattern} -> {safe_solution}")
            lesson_ids.append(le.id)

        if lesson_lines:
            parts.append("[LESSONS FROM PAST FAILURES]")
            parts.extend(lesson_lines)

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
            ValueError: If *outcome* is outside ``[0, 1]`` or NaN.
        """
        # A4: validate outcome BEFORE any side effects.
        if outcome is not None:
            import math
            if math.isnan(outcome) or not (0.0 <= outcome <= 1.0):
                raise ValueError(
                    f"outcome must be in [0, 1] and not NaN, got {outcome}"
                )

        if isinstance(inj_or_handle_id, Injection):
            handle_id = inj_or_handle_id.handle_id
        else:
            handle_id = inj_or_handle_id

        # A2: cross-process file lock wraps the entire check -> reinforce ->
        # trust -> append path, preventing two processes from doubling up.
        # The threading lock is kept outside as an additional in-process gate.
        with self._report_lock, self._writer.lock(self._injections_path):
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
    # Shadow mode (counterfactual graduation)
    # ------------------------------------------------------------------

    def enqueue_counterfactual(
        self,
        inj: Injection,
        baseline_prompt: str,
    ) -> str:
        """Queue a counterfactual trial for a shadow injection.

        Pass the :class:`Injection` from ``inject_for(..., mode="shadow")`` and
        the prompt you would have used WITHOUT it.  Your worker later re-runs
        both against the model, scores each with the surface's deterministic
        validator, and calls :meth:`report_counterfactual_outcome`.

        Args:
            inj: A shadow injection (its ``lesson_ids`` are the lessons under
                test; its ``text`` is appended to *baseline_prompt* to form the
                injected prompt).
            baseline_prompt: The prompt without the lesson block.

        Returns:
            A ``cf_id`` for :meth:`report_counterfactual_outcome`.
        """
        injected_prompt = (
            f"{baseline_prompt}\n{inj.text}" if inj.text else baseline_prompt
        )
        return self._cf_queue.enqueue(
            lesson_ids=inj.lesson_ids,
            model=inj.model,
            baseline_prompt=baseline_prompt,
            injected_prompt=injected_prompt,
        )

    def report_counterfactual_outcome(
        self,
        cf_id: str,
        baseline_passed: bool,
        injected_passed: bool,
    ) -> bool:
        """Close a counterfactual trial and update per-lesson graduation state.

        Updates each lesson's shadow counters (see
        :meth:`LessonStore.record_counterfactual`) so its graduation verdict can
        move toward promote / hold / quarantine.

        **Idempotent**: reporting the same ``cf_id`` twice returns ``False``.

        Args:
            cf_id: Handle from :meth:`enqueue_counterfactual`.
            baseline_passed: Did the validator pass the baseline output?
            injected_passed: Did the validator pass the injected output?

        Returns:
            ``True`` on first report, ``False`` on duplicate.

        Raises:
            KeyError: If *cf_id* was never queued.
        """
        lesson_ids = self._cf_queue.resolve(cf_id, baseline_passed, injected_passed)
        if lesson_ids is None:
            return False
        for lid in lesson_ids:
            self._store.record_counterfactual(lid, baseline_passed, injected_passed)
        return True

    def pending_counterfactuals(self) -> list[Any]:
        """Return counterfactual trials queued but not yet reported."""
        return self._cf_queue.pending()

    def graduation_status(self, lesson_id: str) -> str | None:
        """Graduation verdict for a lesson from its counterfactual evidence.

        Returns ``"promote"``, ``"hold"``, ``"quarantine"``, or ``None`` if the
        lesson is unknown (see :mod:`errlore.lessons.graduation`).
        """
        return self._store.graduation_status(lesson_id)

    def graduated_lessons(self) -> list[Any]:
        """Return lessons whose counterfactual evidence says ``promote``.

        These have cleared the safety bar and shown at least one verified fix;
        they are ready to graduate into a permanent surface (system prompt,
        conventions doc, a PR) with their counts as the evidence trail.
        """
        return self._store.graduated_lessons()

    # ------------------------------------------------------------------
    # Lesson convenience API
    # ------------------------------------------------------------------

    def add_lesson(
        self,
        pattern: str,
        solution: str,
        *,
        task_type: str = "",
        confidence: float = 0.8,
    ) -> str | None:
        """Add a lesson directly (without an error/resolve cycle).

        The *pattern* is sanitized via
        :func:`~errlore.sanitize.sanitize_lesson_text`.  If sanitization
        rejects the text, ``None`` is returned and no lesson is stored.

        Args:
            pattern: Problem pattern description.
            solution: How to fix / avoid the problem.
            task_type: Optional task category for narrower search later.
            confidence: Initial confidence (default 0.8).

        Returns:
            Lesson ID, or ``None`` if the pattern was rejected by the sanitizer.
        """
        sanitized = sanitize_lesson_text(pattern)
        if sanitized is None:
            logger.warning("add_lesson: pattern rejected by sanitizer")
            return None
        return self._store.log_lesson(
            pattern=sanitized,
            solution=solution,
            task_type=task_type,
            confidence=confidence,
        )

    def lessons(self, limit: int | None = None) -> list[Any]:
        """Return all lessons, optionally capped at *limit*.

        Args:
            limit: Maximum number of lessons to return (newest first by
                confidence). ``None`` returns all.

        Returns:
            List of :class:`~errlore.lessons.models.Lesson` objects.
        """
        all_lessons = self._store._read_lessons()
        all_lessons.sort(key=lambda le: le.confidence, reverse=True)
        if limit is not None:
            return all_lessons[:limit]
        return all_lessons

    def check_consistency(
        self,
        outputs: list[str],
        *,
        mode: str = "final_line",
        similarity: float = 1.0,
        model: str | None = None,
        task_type: str = "",
    ) -> Any:
        """Flag likely-wrong output via re-run consistency (warning tier).

        Thin wrapper over :func:`errlore.consistency.check_consistency` for
        validator-less surfaces.  When *model* is given and the outputs are
        unstable, the instability is also recorded via :meth:`log_error`, so a
        recurring flaky surface becomes a tracked failure you can resolve into
        a lesson.

        See :mod:`errlore.consistency` for the honest one-sided operating
        profile (86% precision, ~19% recall; a stable result is not
        verification).

        Args:
            outputs: Two or more independent runs of the same prompt.
            mode: ``"final_line"`` (default) or ``"full"``.
            similarity: Equivalence threshold in ``(0, 1]`` (1.0 = strict).
            model: If set, an unstable verdict is logged as an error for it.
            task_type: Task category for the logged error.

        Returns:
            :class:`~errlore.consistency.ConsistencyResult`.
        """
        from errlore.consistency import check_consistency

        result = check_consistency(outputs, mode=mode, similarity=similarity)
        if model is not None and not result.stable:
            self.log_error(
                model,
                task_type or "consistency",
                error="unstable output across re-runs",
                message=(
                    f"consistency flag: {result.distinct} distinct answers in "
                    f"{result.n_runs} runs (agreement {result.agreement:.2f})"
                ),
            )
        return result

    def quarantined_lessons(self) -> list[Any]:
        """Return lessons the harm gate currently withholds from injection.

        Empty when ``harm_gate=False`` semantics are desired at read time is
        not implied -- this always reflects the harm-gate verdict on each
        lesson's failure history, regardless of the constructor flag, so it
        works as an audit view.

        Returns:
            List of :class:`~errlore.lessons.models.Lesson` objects that are
            quarantined (see :mod:`errlore.lessons.graduation`).
        """
        return self._store.quarantined_lessons()

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
        ``lessons_total``, ``lessons_applied``, ``lessons_quarantined``,
        ``pending_injections``, and (when trust is enabled) ``trust`` — a
        dict of model weights.
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
