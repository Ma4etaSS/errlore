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
