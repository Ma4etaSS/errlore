"""Tests for AgentMemory facade (Phase 2 DoD) and sanitize module."""

from __future__ import annotations

from pathlib import Path

import pytest

from errlore.facade import AgentMemory, Injection
from errlore.sanitize import sanitize_lesson_text

# ===================================================================
# sanitize_lesson_text — unit tests
# ===================================================================


class TestSanitizeLessonText:
    def test_normal_text_passes(self) -> None:
        result = sanitize_lesson_text("Always validate input before processing")
        assert result == "Always validate input before processing"

    def test_empty_returns_none(self) -> None:
        assert sanitize_lesson_text("") is None
        assert sanitize_lesson_text("   ") is None

    def test_raw_json_rejected(self) -> None:
        assert sanitize_lesson_text('{"key": "value", "nested": {"x": 1}}') is None

    def test_raw_json_with_message_extracted(self) -> None:
        text = '{"error": "connection timeout", "code": 500}'
        result = sanitize_lesson_text(text)
        assert result == "connection timeout"

    def test_json_array_rejected(self) -> None:
        assert sanitize_lesson_text('[{"a": 1}, {"b": 2}]') is None

    def test_code_fence_only_rejected(self) -> None:
        text = "```python\ndef foo():\n    return 42\n```"
        assert sanitize_lesson_text(text) is None

    def test_code_fence_with_prose_kept(self) -> None:
        text = (
            "Always check return values:\n"
            "```python\nassert result is not None\n```\n"
            "This prevents crashes."
        )
        result = sanitize_lesson_text(text)
        assert result is not None
        assert "check return values" in result
        assert "prevents crashes" in result
        assert "```" not in result

    def test_whitespace_collapsed(self) -> None:
        text = "Check   the   input   \n\n   before   processing"
        result = sanitize_lesson_text(text)
        assert result == "Check the input before processing"

    def test_truncation_at_word_boundary(self) -> None:
        text = "word " * 100  # 500 chars
        result = sanitize_lesson_text(text, max_len=50)
        assert result is not None
        assert len(result) <= 50
        assert result.endswith("...")

    def test_short_text_no_truncation(self) -> None:
        text = "Keep it short"
        result = sanitize_lesson_text(text, max_len=300)
        assert result == "Keep it short"

    def test_json_with_description_field(self) -> None:
        text = '{"description": "rate limit exceeded", "status": 429}'
        result = sanitize_lesson_text(text)
        assert result == "rate limit exceeded"


# ===================================================================
# AgentMemory facade — integration tests
# ===================================================================


def test_full_learning_cycle(data_dir: Path) -> None:
    """Full DoD: log -> resolve -> inject -> report -> verify."""
    mem = AgentMemory(data_dir)

    # 1. Log error.
    err_id = mem.log_error(
        "gpt-4o", "extraction", "TimeoutError: date extraction failed",
    )
    assert err_id

    # 2. Resolve with lesson.
    ok = mem.resolve(
        err_id,
        resolution="Added retry with backoff",
        lesson="For date extraction, always validate ISO-8601 format",
    )
    assert ok is True

    # 3. Inject for similar task — lesson must appear.
    inj = mem.inject_for(
        "extract dates from contract", "gpt-4o", task_type="extraction",
    )
    assert "[LESSONS FROM PAST FAILURES]" in inj.text
    assert inj.handle_id
    pending = mem.pending_injections()
    assert any(p.handle_id == inj.handle_id for p in pending)

    # 4. Report success.
    result = mem.report_outcome(inj, success=True)
    assert result is True

    # 5. Verify: lesson reinforced.
    lessons = mem._store.search_lessons(task_type="extraction")
    assert lessons
    lesson = lessons[0]
    assert lesson.applied_count == 1
    assert lesson.confidence == 0.9  # 0.8 + 0.1

    # Trust weight shifted up from default 0.5.
    assert mem._trust is not None
    trust_w = mem._trust.get_weight("gpt-4o")
    assert trust_w > 0.5

    # Pending is now empty.
    assert len(mem.pending_injections()) == 0

    # 6. Idempotent: second report returns False.
    result2 = mem.report_outcome(inj, success=True)
    assert result2 is False


def test_negative_outcome_decreases_confidence(data_dir: Path) -> None:
    """report_outcome(success=False) lowers lesson confidence."""
    mem = AgentMemory(data_dir)

    err_id = mem.log_error("gpt-4o", "code_gen", "SyntaxError: invalid code")
    mem.resolve(err_id, "Fixed prompt", lesson="Always include type hints")

    inj = mem.inject_for("generate python code", "gpt-4o", task_type="code_gen")
    mem.report_outcome(inj, success=False)

    lessons = mem._store.search_lessons(task_type="code_gen")
    assert lessons
    assert lessons[0].confidence == 0.7  # 0.8 - 0.1


def test_inject_for_empty(data_dir: Path) -> None:
    """No lessons or errors -> text is empty, handle is still valid."""
    mem = AgentMemory(data_dir)
    inj = mem.inject_for("some task", "gpt-4o")

    assert inj.text == ""
    assert inj.handle_id
    assert inj.lesson_ids == []

    # Can still report on empty injection.
    assert mem.report_outcome(inj, success=True) is True


def test_persistence_across_instances(data_dir: Path) -> None:
    """Handles survive process restart (new instance, same data_dir)."""
    mem1 = AgentMemory(data_dir)
    err_id = mem1.log_error("claude", "research", "ValueError: bad data")
    mem1.resolve(err_id, "Validated input", lesson="Always sanitize input data")
    inj = mem1.inject_for("research task", "claude", task_type="research")

    # New instance on same data_dir.
    mem2 = AgentMemory(data_dir)
    pending = mem2.pending_injections()
    assert any(p.handle_id == inj.handle_id for p in pending)

    # Report from new instance using string handle_id.
    assert mem2.report_outcome(inj.handle_id, success=True) is True
    assert len(mem2.pending_injections()) == 0


def test_resolve_junk_lesson(data_dir: Path) -> None:
    """Raw JSON lesson text is rejected; error still resolves."""
    mem = AgentMemory(data_dir)
    err_id = mem.log_error("gpt-4o", "extraction", "SomeError: test")

    ok = mem.resolve(
        err_id,
        resolution="handled",
        lesson='{"key": "value", "nested": {"x": 1}}',
    )
    assert ok is True  # error resolved

    # But no lesson was created.
    lessons = mem._store.search_lessons(task_type="extraction")
    assert len(lessons) == 0


def test_lazy_decay(data_dir: Path) -> None:
    """decay_every=2 fires decay_unused on the second inject_for call."""
    mem = AgentMemory(data_dir, decay_every=2)

    # Create a lesson with applied_count=0.
    err_id = mem.log_error("gpt-4o", "other_task", "SomeError: test")
    mem.resolve(err_id, "fixed", lesson="Some lesson for other task")

    # First inject_for: counter 0->1, no decay yet.
    mem.inject_for("unrelated", "claude", task_type="unrelated")

    lessons = mem._store.search_lessons(task_type="other_task")
    assert lessons[0].confidence == 0.8

    # Second inject_for: counter 1->2, triggers decay.
    mem.inject_for("still unrelated", "claude", task_type="unrelated")

    lessons = mem._store.search_lessons(task_type="other_task")
    assert lessons
    assert lessons[0].confidence == 0.75  # 0.8 - 0.05


def test_decay_counter_persists_across_restart(data_dir: Path) -> None:
    """Decay fires even when each inject_for runs in a fresh instance.

    Models the flagship Claude Code hook: one short-lived process per call.
    An in-process counter would reset every time and never reach the bar.
    """
    seed = AgentMemory(data_dir, decay_every=2)
    err_id = seed.log_error("gpt-4o", "other_task", "SomeError: test")
    seed.resolve(err_id, "fixed", lesson="Some lesson for other task")

    # First call in one "process": counter 0->1, no decay.
    AgentMemory(data_dir, decay_every=2).inject_for(
        "unrelated", "claude", task_type="unrelated"
    )
    lessons = AgentMemory(data_dir)._store.search_lessons(task_type="other_task")
    assert lessons[0].confidence == 0.8

    # Second call in a brand-new instance: persisted counter 1->2, decay fires.
    AgentMemory(data_dir, decay_every=2).inject_for(
        "still unrelated", "claude", task_type="unrelated"
    )
    lessons = AgentMemory(data_dir)._store.search_lessons(task_type="other_task")
    assert lessons[0].confidence == 0.75  # 0.8 - 0.05


def test_injections_compaction_drops_closed_issued_keeps_pending(
    data_dir: Path,
) -> None:
    """Compaction removes issued records of closed handles, keeps markers +
    pending, and preserves idempotency (a duplicate report still returns False).
    """
    import json

    mem = AgentMemory(data_dir, trust=False)
    mem._COMPACT_THRESHOLD = 4  # force compaction quickly

    err = mem.log_error("m", "t", "Boom: x")
    mem.resolve(err, "fixed", lesson="when boom, check fuse")

    # Two injections we will close, one we leave pending.
    closed = [mem.inject_for("boom task", "m", task_type="t") for _ in range(2)]
    pending = mem.inject_for("boom task", "m", task_type="t")
    for inj in closed:
        assert mem.report_outcome(inj, success=True) is True

    path = data_dir / "injections.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    issued_ids = {r["handle_id"] for r in rows if r["event"] == "issued"}
    reported_ids = {r["handle_id"] for r in rows if r["event"] == "reported"}

    # Closed handles: issued dropped, marker kept. Pending: issued retained.
    for inj in closed:
        assert inj.handle_id not in issued_ids
        assert inj.handle_id in reported_ids
    assert pending.handle_id in issued_ids

    # Idempotency survives compaction: re-reporting a closed handle -> False.
    assert mem.report_outcome(closed[0], success=True) is False
    # Pending is still reportable and still discoverable.
    assert {i.handle_id for i in mem.pending_injections()} == {pending.handle_id}
    assert mem.report_outcome(pending, success=True) is True


def test_trust_disabled(data_dir: Path) -> None:
    """trust=False: everything works, no crash on report, no trust in stats."""
    mem = AgentMemory(data_dir, trust=False)

    err_id = mem.log_error("gpt-4o", "extraction", "SomeError: test")
    mem.resolve(err_id, "handled", lesson="Always validate input")

    inj = mem.inject_for("validate data", "gpt-4o", task_type="extraction")
    assert mem.report_outcome(inj, success=True) is True

    stats = mem.stats()
    assert "trust" not in stats


def test_known_issues_after_repeated_errors(data_dir: Path) -> None:
    """KNOWN ISSUES section appears after 3+ same-type model errors."""
    mem = AgentMemory(data_dir)

    # Log 3 identical errors.
    for _ in range(3):
        mem.log_error(
            "gpt-4o", "extraction", "TimeoutError: connection timed out",
        )

    inj = mem.inject_for(
        "extract data", "gpt-4o", task_type="extraction",
    )
    assert "KNOWN ISSUES" in inj.text
    assert "TimeoutError" in inj.text


def test_report_unknown_handle_raises(data_dir: Path) -> None:
    """Unknown handle_id raises KeyError."""
    mem = AgentMemory(data_dir)
    with pytest.raises(KeyError, match="Unknown injection handle"):
        mem.report_outcome("nonexistent_handle", success=True)


def test_log_error_with_exception(data_dir: Path) -> None:
    """log_error accepts BaseException objects."""
    mem = AgentMemory(data_dir)

    try:
        raise ValueError("bad input value")
    except ValueError as exc:
        err_id = mem.log_error("gpt-4o", "code_gen", exc)

    assert err_id
    errors = mem._store._read_errors()
    assert any(
        e.id == err_id and e.error_type == "ValueError" for e in errors
    )


def test_stats_aggregation(data_dir: Path) -> None:
    """stats() returns correct counts and pending info."""
    mem = AgentMemory(data_dir)
    err_id = mem.log_error("gpt-4o", "extraction", "TestError: test")
    mem.resolve(err_id, "fixed", lesson="Test lesson text here")
    inj = mem.inject_for("test task", "gpt-4o", task_type="extraction")

    stats = mem.stats()
    assert stats["errors_total"] == 1
    assert stats["errors_resolved"] == 1
    assert stats["lessons_total"] == 1
    assert stats["pending_injections"] == 1
    assert "trust" in stats

    mem.report_outcome(inj, success=True)
    stats = mem.stats()
    assert stats["pending_injections"] == 0


def test_model_penalty_grows_with_errors(data_dir: Path) -> None:
    """model_penalty delegates to WarningInjector.get_penalty."""
    mem = AgentMemory(data_dir)

    # No errors: penalty is 0.
    assert mem.model_penalty("gpt-4o", "extraction") == 0.0

    # Add errors.
    for _ in range(5):
        mem.log_error("gpt-4o", "extraction", "TimeoutError: test")

    assert mem.model_penalty("gpt-4o", "extraction") > 0.0


def test_injection_dataclass_fields(data_dir: Path) -> None:
    """Injection carries expected fields."""
    mem = AgentMemory(data_dir)
    inj = mem.inject_for("task", "model", domain="code")
    assert isinstance(inj, Injection)
    assert inj.model == "model"
    assert inj.domain == "code"
    assert inj.created_at  # non-empty ISO timestamp


def test_best_model_returns_highest_trust(data_dir: Path) -> None:
    """best_model returns the model with the highest trust weight."""
    mem = AgentMemory(data_dir)

    # Seed two models with different outcomes
    err1 = mem.log_error("model-a", "task", error="bad")
    mem.resolve(err1, "fixed")
    inj1 = mem.inject_for("task", "model-a")
    mem.report_outcome(inj1, success=True)

    err2 = mem.log_error("model-b", "task", error="bad")
    mem.resolve(err2, "fixed")
    inj2 = mem.inject_for("task", "model-b")
    mem.report_outcome(inj2, success=False)

    best = mem.best_model()
    assert best == "model-a"


def test_best_model_none_when_trust_disabled(data_dir: Path) -> None:
    """best_model returns None when trust is disabled."""
    mem = AgentMemory(data_dir, trust=False)
    assert mem.best_model() is None


def test_best_model_none_when_no_models(data_dir: Path) -> None:
    """best_model returns None when no models registered."""
    mem = AgentMemory(data_dir)
    assert mem.best_model() is None


def test_trust_property_accessible(data_dir: Path) -> None:
    """trust property exposes TrustEngine or None."""
    mem_with = AgentMemory(data_dir)
    assert mem_with.trust is not None

    mem_without = AgentMemory(data_dir / "no_trust", trust=False)
    assert mem_without.trust is None


# ===================================================================
# A2: Cross-process idempotent report_outcome
# ===================================================================


def test_two_instances_report_outcome_once(data_dir: Path) -> None:
    """Two AgentMemory instances on the same data_dir: only one reinforce."""
    mem1 = AgentMemory(data_dir)
    err_id = mem1.log_error("gpt-4o", "task", "SomeError: test")
    mem1.resolve(err_id, "fixed", lesson="Lesson for cross-process test")
    inj = mem1.inject_for("task", "gpt-4o", task_type="task")

    mem2 = AgentMemory(data_dir)

    # Both try to report the same handle.
    r1 = mem1.report_outcome(inj.handle_id, success=True)
    r2 = mem2.report_outcome(inj.handle_id, success=True)

    # Exactly one succeeds.
    assert r1 is True
    assert r2 is False

    # Lesson reinforced exactly once.
    lessons = mem1._store.search_lessons(task_type="task")
    assert lessons[0].applied_count == 1


# ===================================================================
# A4: Validate outcome before side effects
# ===================================================================


def test_invalid_outcome_raises_before_reinforce(data_dir: Path) -> None:
    """outcome=2.5 -> ValueError; applied_count unchanged, no reported event."""
    mem = AgentMemory(data_dir)
    err_id = mem.log_error("gpt-4o", "task", "SomeError: test")
    mem.resolve(err_id, "fixed", lesson="Lesson for validation test")
    inj = mem.inject_for("task", "gpt-4o", task_type="task")

    with pytest.raises(ValueError, match="outcome"):
        mem.report_outcome(inj, success=True, outcome=2.5)

    # No side effects occurred.
    lessons = mem._store.search_lessons(task_type="task")
    assert lessons[0].applied_count == 0

    # No "reported" event written.
    events = mem._writer.read_all(mem._injections_path)
    reported = [e for e in events if e.get("event") == "reported"]
    assert len(reported) == 0


def test_nan_outcome_raises(data_dir: Path) -> None:
    """NaN outcome raises ValueError."""
    import math
    mem = AgentMemory(data_dir)
    inj = mem.inject_for("task", "model")
    with pytest.raises(ValueError, match="outcome"):
        mem.report_outcome(inj, success=True, outcome=math.nan)


# ===================================================================
# C4: add_lesson / lessons convenience API
# ===================================================================


def test_add_lesson_and_list(data_dir: Path) -> None:
    """add_lesson stores a lesson; lessons() returns it."""
    mem = AgentMemory(data_dir)
    lid = mem.add_lesson(
        "Always validate input before sending to LLM",
        "Use pydantic BaseModel",
        task_type="validation",
    )
    assert lid is not None

    all_lessons = mem.lessons()
    assert len(all_lessons) == 1
    assert all_lessons[0].id == lid


def test_add_lesson_rejects_junk(data_dir: Path) -> None:
    """add_lesson returns None for raw JSON patterns."""
    mem = AgentMemory(data_dir)
    result = mem.add_lesson('{"key": "value"}', "no solution needed")
    assert result is None
    assert len(mem.lessons()) == 0


def test_lessons_limit(data_dir: Path) -> None:
    """lessons(limit=N) returns at most N lessons."""
    mem = AgentMemory(data_dir)
    for i in range(5):
        mem.add_lesson(f"Unique pattern number {i} for limiting", f"Solution {i}")
    assert len(mem.lessons(limit=3)) == 3
    assert len(mem.lessons()) == 5
