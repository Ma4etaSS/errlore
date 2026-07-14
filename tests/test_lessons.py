"""Tests for errlore.lessons subsystem."""

from __future__ import annotations

from pathlib import Path

import pytest

from errlore.lessons import LessonStore

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def store(data_dir: Path) -> LessonStore:
    """LessonStore backed by the isolated tmp data_dir."""
    return LessonStore(data_dir)


# ------------------------------------------------------------------
# log_error -> resolve_error -> auto-lesson
# ------------------------------------------------------------------


class TestErrorToLesson:
    """log_error -> resolve with lesson -> lesson appears with source link."""

    def test_resolve_creates_linked_lesson(self, store: LessonStore) -> None:
        error_id = store.log_error(
            model="gpt-4o",
            task_type="code",
            error_type="TypeError",
            message="NoneType has no attribute 'strip'",
        )
        assert len(error_id) == 12

        ok = store.resolve_error(
            error_id,
            resolution="Added None check before .strip()",
            lesson="Always guard .strip() calls with an is-None check",
        )
        assert ok is True

        lessons = store.search_lessons(error_type="TypeError")
        assert len(lessons) >= 1
        lesson = lessons[0]
        assert lesson.source_error_id == error_id
        assert lesson.confidence == 0.8
        assert lesson.error_type == "TypeError"
        assert "strip" in lesson.pattern.lower()

    def test_resolve_without_lesson_flag(self, store: LessonStore) -> None:
        error_id = store.log_error(
            model="sonnet",
            task_type="analysis",
            error_type="Timeout",
            message="Model took too long",
        )
        store.resolve_error(error_id, resolution="Increased timeout")
        # No lesson should be created
        lessons = store.search_lessons(query="timeout")
        assert len(lessons) == 0

    def test_resolve_nonexistent_error(self, store: LessonStore) -> None:
        ok = store.resolve_error("doesnotexist", resolution="nope")
        assert ok is False

    def test_resolve_already_resolved(self, store: LessonStore) -> None:
        eid = store.log_error(
            model="m", task_type="t", error_type="E", message="msg",
        )
        store.resolve_error(eid, resolution="first")
        # Second resolve is idempotent
        assert store.resolve_error(eid, resolution="second") is True


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------


class TestDeduplication:
    """Exact and fuzzy dedup of lessons."""

    def test_exact_dedup(self, store: LessonStore) -> None:
        id1 = store.log_lesson(pattern="Always check None", solution="use `if x`")
        id2 = store.log_lesson(pattern="always check none", solution="different wording")
        assert id1 == id2
        # Only one record on disk
        assert store.counts()["lessons_total"] == 1

    def test_fuzzy_dedup_above_85_percent(self, store: LessonStore) -> None:
        id1 = store.log_lesson(
            pattern="Always validate user input data before processing",
            solution="sol1",
        )
        # 6/7 word overlap = 85.7% > 85% threshold
        id2 = store.log_lesson(
            pattern="Always validate user input data before parsing",
            solution="sol2",
        )
        assert id1 == id2
        assert store.counts()["lessons_total"] == 1

    def test_no_dedup_below_threshold(self, store: LessonStore) -> None:
        id1 = store.log_lesson(pattern="Handle timeout errors gracefully", solution="s1")
        id2 = store.log_lesson(pattern="Compress images before upload", solution="s2")
        assert id1 != id2
        assert store.counts()["lessons_total"] == 2


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


class TestSearch:
    """Structured + fuzzy search, confidence ranking."""

    def test_search_by_task_type(self, store: LessonStore) -> None:
        store.log_lesson(pattern="p1", solution="s1", task_type="code")
        store.log_lesson(pattern="p2", solution="s2", task_type="analysis")
        store.log_lesson(pattern="p3", solution="s3", task_type="code")

        results = store.search_lessons(task_type="code")
        assert len(results) == 2
        assert all(r.task_type == "code" for r in results)

    def test_search_by_error_type(self, store: LessonStore) -> None:
        store.log_lesson(pattern="p1", solution="s1", error_type="TypeError")
        store.log_lesson(pattern="p2", solution="s2", error_type="ValueError")

        results = store.search_lessons(error_type="TypeError")
        assert len(results) == 1
        assert results[0].error_type == "TypeError"

    def test_search_fuzzy_query(self, store: LessonStore) -> None:
        store.log_lesson(
            pattern="Always validate user input before processing",
            solution="use pydantic",
        )
        results = store.search_lessons(query="validate input processing")
        assert len(results) >= 1

    def test_search_substring_fallback(self, store: LessonStore) -> None:
        store.log_lesson(pattern="Handle ChromaDB connection drops", solution="retry")
        results = store.search_lessons(query="ChromaDB")
        assert len(results) >= 1
        assert "ChromaDB" in results[0].pattern

    def test_search_ranked_by_confidence(self, store: LessonStore) -> None:
        store.log_lesson(
            pattern="Retry on transient network failure",
            solution="s1",
            confidence=0.5,
            task_type="infra",
        )
        store.log_lesson(
            pattern="Cache DNS lookups for stability",
            solution="s2",
            confidence=0.95,
            task_type="infra",
        )
        results = store.search_lessons(task_type="infra")
        assert len(results) == 2
        assert results[0].confidence > results[1].confidence

    def test_search_empty_params_returns_nothing(self, store: LessonStore) -> None:
        store.log_lesson(pattern="anything", solution="s")
        assert store.search_lessons() == []

    def test_search_respects_limit(self, store: LessonStore) -> None:
        # Patterns must differ enough to avoid fuzzy dedup (< 85% word overlap)
        patterns = [
            "Handle database connection timeout errors gracefully",
            "Compress uploaded images to reduce storage costs",
            "Retry failed API calls with exponential backoff strategy",
            "Validate configuration files before application startup",
            "Cache frequently accessed queries for better performance",
        ]
        for i, pat in enumerate(patterns):
            store.log_lesson(pattern=pat, solution=f"solution {i}", task_type="bulk")
        results = store.search_lessons(task_type="bulk", limit=3)
        assert len(results) == 3


# ------------------------------------------------------------------
# Reinforce
# ------------------------------------------------------------------


class TestReinforce:
    """reinforce(lesson_id, success) adjusts confidence and applied_count."""

    def test_reinforce_success(self, store: LessonStore) -> None:
        lid = store.log_lesson(pattern="p", solution="s", confidence=0.5)
        ok = store.reinforce(lid, success=True)
        assert ok is True

        lessons = store.search_lessons(query="p")
        assert len(lessons) == 1
        assert lessons[0].confidence == pytest.approx(0.6)
        assert lessons[0].applied_count == 1

    def test_reinforce_failure(self, store: LessonStore) -> None:
        lid = store.log_lesson(pattern="p", solution="s", confidence=0.5)
        store.reinforce(lid, success=False)

        lessons = store.search_lessons(query="p")
        assert lessons[0].confidence == pytest.approx(0.4)
        assert lessons[0].applied_count == 1

    def test_reinforce_clamp_upper(self, store: LessonStore) -> None:
        lid = store.log_lesson(pattern="p", solution="s", confidence=0.95)
        store.reinforce(lid, success=True)

        lessons = store.search_lessons(query="p")
        assert lessons[0].confidence == 1.0

    def test_reinforce_clamp_lower(self, store: LessonStore) -> None:
        lid = store.log_lesson(pattern="p", solution="s", confidence=0.15)
        store.reinforce(lid, success=False)

        lessons = store.search_lessons(query="p")
        assert lessons[0].confidence == 0.1

    def test_reinforce_nonexistent(self, store: LessonStore) -> None:
        ok = store.reinforce("doesnotexist", success=True)
        assert ok is False

    def test_reinforce_is_persistent(self, store: LessonStore, data_dir: Path) -> None:
        """Reinforced state survives a new LessonStore instance."""
        lid = store.log_lesson(pattern="p", solution="s", confidence=0.5)
        store.reinforce(lid, success=True)

        store2 = LessonStore(data_dir)
        lessons = store2.search_lessons(query="p")
        assert lessons[0].confidence == pytest.approx(0.6)
        assert lessons[0].applied_count == 1


# ------------------------------------------------------------------
# Decay
# ------------------------------------------------------------------


class TestDecay:
    """decay_unused touches only unused lessons."""

    def test_decay_unused_only(self, store: LessonStore) -> None:
        lid_used = store.log_lesson(
            pattern="used lesson about validation",
            solution="s1",
            confidence=0.8,
        )
        store.reinforce(lid_used, success=True)  # applied_count=1

        store.log_lesson(
            pattern="unused lesson about caching",
            solution="s2",
            confidence=0.8,
        )

        decayed = store.decay_unused()
        assert decayed == 1

        lessons = store.search_lessons(query="validation")
        assert lessons[0].confidence == pytest.approx(0.9)  # reinforced, not decayed

        unused = store.search_lessons(query="caching")
        assert unused[0].confidence == pytest.approx(0.75)

    def test_decay_skips_low_confidence(self, store: LessonStore) -> None:
        store.log_lesson(pattern="low conf lesson topic", solution="s", confidence=0.25)
        decayed = store.decay_unused()
        assert decayed == 0
        lessons = store.search_lessons(query="low conf lesson topic")
        assert lessons[0].confidence == pytest.approx(0.25)

    def test_decay_floor_at_zero(self, store: LessonStore) -> None:
        store.log_lesson(
            pattern="floor test lesson about edges",
            solution="s",
            confidence=0.32,
        )
        store.decay_unused()  # 0.32 > 0.3 so it decays
        lessons = store.search_lessons(query="floor test lesson about edges")
        assert lessons[0].confidence >= 0.0


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


class TestPersistence:
    """New LessonStore instance sees previously written data."""

    def test_errors_persist(self, store: LessonStore, data_dir: Path) -> None:
        store.log_error(
            model="m", task_type="t", error_type="E", message="msg",
        )
        store2 = LessonStore(data_dir)
        stats = store2.counts()
        assert stats["errors_total"] == 1
        assert stats["errors_unresolved"] == 1

    def test_lessons_persist(self, store: LessonStore, data_dir: Path) -> None:
        store.log_lesson(pattern="persist test", solution="s")
        store2 = LessonStore(data_dir)
        results = store2.search_lessons(query="persist test")
        assert len(results) == 1


# ------------------------------------------------------------------
# Concurrency safety (atomic_rewrite)
# ------------------------------------------------------------------


class TestConcurrency:
    """reinforce is atomic-rewrite safe under concurrent calls."""

    def test_concurrent_reinforce(self, store: LessonStore) -> None:
        import threading

        lid = store.log_lesson(pattern="concurrent test", solution="s", confidence=0.5)
        errors: list[Exception] = []

        def _reinforce() -> None:
            try:
                store.reinforce(lid, success=True)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_reinforce) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        lessons = store.search_lessons(query="concurrent test")
        assert len(lessons) == 1
        # 10 reinforcements from 0.5: clamped at 1.0
        assert lessons[0].confidence == 1.0
        assert lessons[0].applied_count == 10


# ------------------------------------------------------------------
# Counts / stats
# ------------------------------------------------------------------


class TestCounts:
    """counts() returns correct statistics."""

    def test_empty_store(self, store: LessonStore) -> None:
        stats = store.counts()
        assert stats == {
            "errors_total": 0,
            "errors_resolved": 0,
            "errors_unresolved": 0,
            "lessons_total": 0,
            "lessons_applied": 0,
            "lessons_quarantined": 0,
        }

    def test_mixed_state(self, store: LessonStore) -> None:
        e1 = store.log_error(model="m", task_type="t", error_type="E1", message="m1")
        store.log_error(model="m", task_type="t", error_type="E2", message="m2")
        store.resolve_error(e1, resolution="fixed", lesson="lesson text")

        lid = store.log_lesson(pattern="manual lesson", solution="s")
        store.reinforce(lid, success=True)

        stats = store.counts()
        assert stats["errors_total"] == 2
        assert stats["errors_resolved"] == 1
        assert stats["errors_unresolved"] == 1
        # auto-lesson from resolve + manual lesson = 2
        assert stats["lessons_total"] == 2
        # only the manually reinforced one
        assert stats["lessons_applied"] == 1
