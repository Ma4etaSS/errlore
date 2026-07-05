"""Tests for errlore.errmem — error memory (Amygdala) subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from errlore.errmem.classifier import classify_error
from errlore.errmem.injector import WarningInjector, sanitize_description
from errlore.errmem.patterns import PatternDetector
from errlore.errmem.tracker import ErrorTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_error(
    typ: str = "TimeoutError",
    desc: str = "model took too long",
    severity: str = "high",
) -> dict[str, Any]:
    return {"type": typ, "description": desc, "severity": severity}


def _build_stack(
    tracker: ErrorTracker,
    model: str,
    task_type: str,
    error: dict[str, Any],
    count: int,
) -> None:
    """Record the same error ``count`` times."""
    for _ in range(count):
        tracker.record_error(model, task_type, error)


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_from_exception_object(self) -> None:
        assert classify_error(error=ValueError("bad")) == "ValueError"

    def test_from_message(self) -> None:
        assert classify_error(message="got a TimeoutError after 30s") == "TimeoutError"

    def test_from_stacktrace_last_match(self) -> None:
        trace = (
            "Traceback (most recent call last):\n"
            "  File foo.py, line 1\n"
            "RuntimeError: wrapped\n"
            "  File bar.py, line 2\n"
            "ConnectionError: refused"
        )
        assert classify_error(stacktrace=trace) == "ConnectionError"

    def test_custom_error_from_message(self) -> None:
        assert classify_error(message="MyCustomException raised") == "MyCustomException"

    def test_fallback(self) -> None:
        assert classify_error() == "UnclassifiedError"

    def test_exception_takes_priority(self) -> None:
        result = classify_error(
            error=KeyError("k"),
            message="TimeoutError happened",
            stacktrace="ValueError: x",
        )
        assert result == "KeyError"


# ---------------------------------------------------------------------------
# tracker — record + profile persistence
# ---------------------------------------------------------------------------


class TestTracker:
    def test_record_and_read_back(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir)
        entry = tracker.record_error("gpt-4o", "summarization", _make_error())
        assert entry["model"] == "gpt-4o"
        assert entry["error_type"] == "TimeoutError"

        errors = tracker.get_model_errors("gpt-4o")
        assert len(errors) == 1
        assert errors[0]["task_type"] == "summarization"

    def test_model_profile_persists(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir)
        _build_stack(tracker, "sonnet", "code", _make_error("ValueError", "bad parse"), 4)

        # Recreate tracker from same dir — data must survive
        tracker2 = ErrorTracker(data_dir)
        profile = tracker2.get_model_profile("sonnet")
        assert profile["errors"] == 4
        assert "ValueError (x4)" in profile["weaknesses"]
        assert profile["last_error"] != ""

    def test_model_stats_aggregation(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir)
        _build_stack(tracker, "m1", "t1", _make_error(), 2)
        _build_stack(tracker, "m2", "t2", _make_error("ImportError"), 1)

        stats = tracker.get_model_stats()
        assert stats["m1"]["errors"] == 2
        assert stats["m2"]["errors"] == 1


# ---------------------------------------------------------------------------
# patterns — 3 reps detected, 2 reps not
# ---------------------------------------------------------------------------


class TestPatterns:
    def test_three_reps_detected(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir, min_occurrences=3)
        _build_stack(tracker, "opus", "code", _make_error("TimeoutError"), 3)

        detector = PatternDetector(min_occurrences=3)
        errors = tracker.get_model_errors("opus")
        patterns = detector.detect(errors)
        assert len(patterns) == 1
        assert patterns[0]["occurrences"] == 3

    def test_two_reps_not_detected(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir)
        _build_stack(tracker, "opus", "code", _make_error("TimeoutError"), 2)

        detector = PatternDetector(min_occurrences=3)
        errors = tracker.get_model_errors("opus")
        patterns = detector.detect(errors)
        assert len(patterns) == 0

    def test_custom_threshold(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir)
        _build_stack(tracker, "m", "t", _make_error(), 5)

        detector = PatternDetector(min_occurrences=5)
        patterns = detector.detect(tracker.get_model_errors("m"))
        assert len(patterns) == 1
        assert patterns[0]["occurrences"] == 5


# ---------------------------------------------------------------------------
# sanitize_description
# ---------------------------------------------------------------------------


class TestSanitize:
    def test_normal_text_passes(self) -> None:
        assert sanitize_description("model took too long") == "model took too long"

    def test_empty_returns_none(self) -> None:
        assert sanitize_description("") is None
        assert sanitize_description("   ") is None

    def test_raw_json_discarded(self) -> None:
        blob = '{"response_id": "abc", "tokens": 1500, "latency_ms": 42000}'
        assert sanitize_description(blob) is None

    def test_json_with_message_extracted(self) -> None:
        blob = '{"error": "connection refused", "code": 502}'
        result = sanitize_description(blob)
        assert result == "connection refused"

    def test_backtick_blob_discarded(self) -> None:
        blob = "```json\n{\"key\": \"val\"}\n```"
        assert sanitize_description(blob) is None

    def test_long_text_truncated(self) -> None:
        long = "x" * 300
        result = sanitize_description(long)
        assert result is not None
        assert len(result) == 200
        assert result.endswith("...")

    def test_json_with_message_key(self) -> None:
        blob = '{"message": "Rate limit exceeded for tier", "status": 429}'
        result = sanitize_description(blob)
        assert result == "Rate limit exceeded for tier"

    def test_raw_json_not_in_warning(self, data_dir: Path) -> None:
        """End-to-end: raw JSON descriptions must not leak into warnings."""
        tracker = ErrorTracker(data_dir, min_occurrences=1)
        detector = PatternDetector(min_occurrences=1)
        injector = WarningInjector(tracker, detector)

        # Record error with raw JSON description
        tracker.record_error(
            "gpt-4o",
            "analysis",
            {
                "type": "APIError",
                "description": '{"response_id": "x", "tokens": 99}',
            },
        )

        warning = injector.build_warning("gpt-4o", "analysis")
        # The weakness line "APIError (x1)" is allowed,
        # but the JSON blob must not appear
        assert '{"response_id"' not in warning


# ---------------------------------------------------------------------------
# injector — build_warning format
# ---------------------------------------------------------------------------


class TestInjector:
    def test_empty_when_no_errors(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir)
        detector = PatternDetector()
        injector = WarningInjector(tracker, detector)
        assert injector.build_warning("m", "t") == ""

    def test_known_issues_format(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir, min_occurrences=3)
        detector = PatternDetector(min_occurrences=3)
        injector = WarningInjector(tracker, detector)

        _build_stack(tracker, "opus", "code", _make_error("TimeoutError", "slow"), 4)

        warning = injector.build_warning("opus", "code")
        lines = warning.split("\n")
        assert lines[0] == "KNOWN ISSUES:"
        assert "TimeoutError (x4)" in lines[1]
        assert any("Past error on similar task:" in ln for ln in lines)

    def test_warning_count_matches(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir, min_occurrences=2)
        detector = PatternDetector()
        injector = WarningInjector(tracker, detector)

        _build_stack(tracker, "m", "t", _make_error("E1", "d1"), 5)
        _build_stack(tracker, "m", "t", _make_error("E2", "d2"), 3)

        warning = injector.build_warning("m", "t")
        assert "(x5)" in warning
        assert "(x3)" in warning


# ---------------------------------------------------------------------------
# get_penalty
# ---------------------------------------------------------------------------


class TestPenalty:
    def test_zero_when_no_errors(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir)
        detector = PatternDetector()
        injector = WarningInjector(tracker, detector)
        assert injector.get_penalty("m", "t") == 0.0

    def test_grows_with_errors(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir, min_occurrences=3)
        detector = PatternDetector(min_occurrences=3)
        injector = WarningInjector(tracker, detector)

        _build_stack(tracker, "m", "code", _make_error("E1", "code fails"), 3)
        p1 = injector.get_penalty("m", "code")

        _build_stack(tracker, "m", "code", _make_error("E2", "code breaks"), 3)
        p2 = injector.get_penalty("m", "code")

        assert 0.0 < p1 <= 1.0
        assert 0.0 < p2 <= 1.0
        assert p2 >= p1, "penalty must grow with more error types"

    def test_bounded_at_one(self, data_dir: Path) -> None:
        tracker = ErrorTracker(data_dir, min_occurrences=1)
        detector = PatternDetector(min_occurrences=1)
        injector = WarningInjector(tracker, detector)

        # Record a ton of different error types
        for i in range(50):
            _build_stack(tracker, "m", "t", _make_error(f"E{i}", "t fails"), 2)

        p = injector.get_penalty("m", "t")
        assert p <= 1.0
