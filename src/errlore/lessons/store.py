"""Persistent lesson store backed by JSONL via errlore.io.

Manages two files:
    data_dir/errors.jsonl   -- error events
    data_dir/lessons.jsonl  -- extracted lessons (single-record, mutable via atomic_rewrite)

Unlike the NEXUS original, lessons are updated *in-place* via atomic_rewrite
instead of appending new versions.  This eliminates dedup ambiguity at read time.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from errlore.io import JSONLWriter
from errlore.lessons.models import ErrorRecord, Lesson

if TYPE_CHECKING:
    from errlore.retrieval import LessonRetriever

logger = logging.getLogger("errlore.lessons")


class LessonStore:
    """Thread-safe lesson store persisted as JSONL.

    Args:
        data_dir: Directory for errors.jsonl and lessons.jsonl.
        retriever: Optional semantic retriever implementing
            :class:`~errlore.retrieval.LessonRetriever`.  When provided,
            :meth:`search_lessons` uses semantic search with automatic
            fallback to word-overlap when no semantic results are found.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        retriever: LessonRetriever | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._errors_path = self._data_dir / "errors.jsonl"
        self._lessons_path = self._data_dir / "lessons.jsonl"
        self._writer = JSONLWriter()
        self._lock = threading.Lock()
        self._retriever = retriever

        if self._retriever is not None:
            self._sync_retriever()

    # ------------------------------------------------------------------
    # Retriever sync
    # ------------------------------------------------------------------

    def _sync_retriever(self) -> None:
        """Ensure all existing lessons are indexed in the retriever.

        Called once during construction.  Each ``add`` is idempotent so
        already-indexed lessons are skipped with only a set-lookup cost.
        """
        if self._retriever is None:
            return
        for lesson in self._read_lessons():
            self._retriever.add(lesson.id, lesson.pattern)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def log_error(
        self,
        model: str,
        task_type: str,
        error_type: str,
        message: str,
        *,
        context: str = "",
        stacktrace: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record an error event.

        Returns:
            The generated error ID (12-char hex).
        """
        record = ErrorRecord(
            model=model,
            task_type=task_type,
            error_type=error_type,
            message=message,
            context=context,
            stacktrace=stacktrace,
            metadata=metadata,
        )
        self._writer.append(self._errors_path, record.to_dict())
        logger.debug("Logged error %s: %s", record.id, message[:80])
        return record.id

    def resolve_error(
        self,
        error_id: str,
        resolution: str,
        lesson: str | None = None,
    ) -> bool:
        """Resolve an error by ID.  Optionally extract a lesson automatically.

        When *lesson* is provided the store creates a new lesson entry with
        confidence 0.8, linking it to the resolved error via ``source_error_id``.

        The error record is updated in-place (atomic_rewrite).

        Returns:
            True if the error was found and resolved, False otherwise.
        """
        # Race-safe: read-modify-write happens under ONE file lock via
        # atomic_update, so errors appended concurrently by other threads
        # or processes are never lost (they land before or after, intact).
        target: ErrorRecord | None = None
        already_resolved = False

        def _apply(
            entries: list[dict[str, object]],
        ) -> list[dict[str, object]] | None:
            nonlocal target, already_resolved
            for entry in entries:
                if entry.get("id") == error_id:
                    rec = ErrorRecord.from_dict(entry)
                    if rec.resolved:
                        already_resolved = True
                        target = rec
                        return None  # abort write, nothing to change
                    rec.resolved = True
                    rec.resolution = resolution
                    target = rec
                    entry.update(rec.to_dict())
                    return entries
            return None  # not found, abort write

        with self._lock:
            self._writer.atomic_update(self._errors_path, _apply)

        if target is None:
            logger.debug("resolve_error: error_id=%s not found", error_id)
            return False
        if already_resolved:
            logger.debug("resolve_error: error_id=%s already resolved", error_id)
            return True

        if lesson:
            self.log_lesson(
                pattern=f"{target.error_type}: {target.message}",
                solution=lesson,
                confidence=0.8,
                task_type=target.task_type,
                error_type=target.error_type,
                source_error_id=error_id,
            )

        logger.info("Resolved error %s: %s", error_id, resolution[:80])
        return True

    # ------------------------------------------------------------------
    # Lessons
    # ------------------------------------------------------------------

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        """Word-level Jaccard-style overlap (|intersection| / max(|A|, |B|))."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        return len(intersection) / max(len(words_a), len(words_b))

    def _find_duplicate_lesson(self, pattern: str) -> Lesson | None:
        """Check recent lessons for exact or fuzzy (>85% word overlap) duplicate."""
        lessons = self._read_lessons()
        normalized = pattern.lower().strip()
        for lesson in lessons:
            existing = lesson.pattern.lower().strip()
            if existing == normalized:
                return lesson
            if self._word_overlap(normalized, existing) > 0.85:
                return lesson
        return None

    def log_lesson(
        self,
        pattern: str,
        solution: str,
        *,
        confidence: float = 0.8,
        task_type: str = "",
        error_type: str = "",
        source_error_id: str = "",
        source_errors: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a lesson with deduplication.

        Dedup rules (same as NEXUS source):
        - Exact match on pattern (case-insensitive, stripped)
        - Fuzzy match: word overlap > 85%

        If a duplicate is found the existing lesson ID is returned and
        no new record is written.

        Returns:
            Lesson ID (new or existing duplicate).
        """
        with self._lock:
            existing = self._find_duplicate_lesson(pattern)
            if existing is not None:
                logger.debug(
                    "Duplicate lesson skipped: '%s' matches '%s'",
                    pattern[:60],
                    existing.pattern[:60],
                )
                return existing.id

            lesson = Lesson(
                pattern=pattern,
                solution=solution,
                confidence=confidence,
                task_type=task_type,
                error_type=error_type,
                source_error_id=source_error_id,
                source_errors=source_errors or [],
                metadata=metadata,
            )
            self._writer.append(self._lessons_path, lesson.to_dict())
            if self._retriever is not None:
                self._retriever.add(lesson.id, lesson.pattern)
            logger.debug("Logged lesson %s: %s", lesson.id, pattern[:80])
            return lesson.id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_lessons(
        self,
        *,
        query: str = "",
        task_type: str = "",
        error_type: str = "",
        limit: int = 10,
    ) -> list[Lesson]:
        """Search lessons by structured filters and/or fuzzy text query.

        When a semantic retriever is configured and *query* is non-empty,
        semantic search runs first.  If it yields no results (after
        applying any task_type/error_type filters), the method falls back
        to the original word-overlap logic.

        Filter priority (word-overlap path):
        1. task_type / error_type exact match (both can combine)
        2. query: word overlap > 30%, fallback to substring match
        3. Results sorted by confidence descending

        Returns:
            Up to *limit* lessons.
        """
        if not query and not task_type and not error_type:
            return []

        # -- Semantic path (retriever + query) ----------------------------
        if self._retriever is not None and query:
            sem = self._semantic_search(query, task_type, error_type, limit)
            if sem:
                return sem
            # Fall through to word-overlap below.

        # -- Word-overlap path (original logic) ---------------------------
        return self._word_overlap_search(query, task_type, error_type, limit)

    def _semantic_search(
        self,
        query: str,
        task_type: str,
        error_type: str,
        limit: int,
    ) -> list[Lesson]:
        """Run semantic search via the configured retriever.

        Returns an empty list when no relevant lessons are found so the
        caller can fall back to word-overlap.
        """
        assert self._retriever is not None

        # Fetch more candidates than needed to allow for post-filtering.
        candidates = self._retriever.search(query, k=limit * 3)
        if not candidates:
            return []

        score_map: dict[str, float] = {cid: score for cid, score in candidates}
        candidate_ids = set(score_map)

        all_lessons = self._read_lessons()
        matched: list[Lesson] = []
        for lesson in all_lessons:
            if lesson.id not in candidate_ids:
                continue
            if task_type and lesson.task_type != task_type:
                continue
            if error_type and lesson.error_type != error_type:
                continue
            matched.append(lesson)

        # Rank by semantic score (primary), confidence (secondary).
        matched.sort(
            key=lambda x: (score_map.get(x.id, 0.0), x.confidence),
            reverse=True,
        )
        return matched[:limit]

    def _word_overlap_search(
        self,
        query: str,
        task_type: str,
        error_type: str,
        limit: int,
    ) -> list[Lesson]:
        """Original word-overlap search logic (preserved for fallback)."""
        all_lessons = self._read_lessons()
        results: list[Lesson] = []

        if task_type or error_type:
            for lesson in all_lessons:
                if task_type and lesson.task_type != task_type:
                    continue
                if error_type and lesson.error_type != error_type:
                    continue
                results.append(lesson)
            # If query is also provided, further filter results
            if query and results:
                filtered: list[Lesson] = []
                for lesson in results:
                    overlap = self._word_overlap(query, lesson.pattern)
                    if overlap > 0.3 or query.lower() in lesson.pattern.lower():
                        filtered.append(lesson)
                if filtered:
                    results = filtered
        elif query:
            # Pure fuzzy search
            for lesson in all_lessons:
                overlap = self._word_overlap(query, lesson.pattern)
                if overlap > 0.3:
                    results.append(lesson)
            # Fallback: substring
            if not results:
                query_lower = query.lower()
                for lesson in all_lessons:
                    if query_lower in lesson.pattern.lower():
                        results.append(lesson)

        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Reinforce / Decay
    # ------------------------------------------------------------------

    def reinforce(self, lesson_id: str, success: bool) -> bool:
        """Reinforce a lesson: adjust confidence +/-0.1, increment applied_count.

        Confidence is clamped to [0.1, 1.0].  The lesson record is updated
        in-place via atomic_rewrite (no append of a new version).

        Returns:
            True if the lesson was found and updated, False otherwise.
        """
        from errlore.lessons.models import _utc_now_iso

        # Race-safe read-modify-write under one file lock (see resolve_error).
        target: Lesson | None = None

        def _apply(
            entries: list[dict[str, object]],
        ) -> list[dict[str, object]] | None:
            nonlocal target
            for entry in entries:
                if entry.get("id") == lesson_id:
                    les = Lesson.from_dict(entry)
                    delta = 0.1 if success else -0.1
                    les.confidence = round(
                        max(0.1, min(1.0, les.confidence + delta)), 2
                    )
                    les.applied_count += 1
                    les.updated_at = _utc_now_iso()
                    target = les
                    entry.update(les.to_dict())
                    return entries
            return None  # not found, abort write

        with self._lock:
            self._writer.atomic_update(self._lessons_path, _apply)

        if target is None:
            logger.debug("reinforce: lesson_id=%s not found", lesson_id)
            return False
        # No retriever re-sync needed: reinforce never changes pattern text,
        # so the embedding vector is unchanged.
        logger.debug(
            "Reinforced lesson %s: confidence=%.2f applied_count=%d",
            lesson_id,
            target.confidence,
            target.applied_count,
        )
        return True

    def decay_unused(self) -> int:
        """Decay confidence of unused lessons.

        Reduces confidence by 0.05 for lessons with applied_count == 0
        and confidence > 0.3.  Updated via single atomic_rewrite.

        Returns:
            Number of lessons that were decayed.
        """
        from errlore.lessons.models import _utc_now_iso

        decayed_count = 0

        def _apply(
            entries: list[dict[str, object]],
        ) -> list[dict[str, object]] | None:
            nonlocal decayed_count
            now = _utc_now_iso()
            for entry in entries:
                les = Lesson.from_dict(entry)
                if les.applied_count > 0 or les.confidence <= 0.3:
                    continue
                les.confidence = round(max(0.0, les.confidence - 0.05), 2)
                les.updated_at = now
                entry.update(les.to_dict())
                decayed_count += 1
            if decayed_count == 0:
                return None  # nothing to change, abort write
            return entries

        with self._lock:
            self._writer.atomic_update(self._lessons_path, _apply)

        if decayed_count > 0:
            logger.info("Decayed %d unused lessons", decayed_count)
        return decayed_count

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Return basic statistics about stored errors and lessons.

        Returns:
            Dict with keys: errors_total, errors_resolved, errors_unresolved,
            lessons_total, lessons_applied (applied_count > 0).
        """
        errors = self._read_errors()
        lessons = self._read_lessons()

        resolved = sum(1 for e in errors if e.resolved)
        applied = sum(1 for le in lessons if le.applied_count > 0)

        return {
            "errors_total": len(errors),
            "errors_resolved": resolved,
            "errors_unresolved": len(errors) - resolved,
            "lessons_total": len(lessons),
            "lessons_applied": applied,
        }

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _read_errors(self) -> list[ErrorRecord]:
        """Read all error records from disk."""
        raw = self._writer.read_all(self._errors_path)
        return [ErrorRecord.from_dict(r) for r in raw]

    def _read_lessons(self) -> list[Lesson]:
        """Read all lesson records from disk."""
        raw = self._writer.read_all(self._lessons_path)
        return [Lesson.from_dict(r) for r in raw]
