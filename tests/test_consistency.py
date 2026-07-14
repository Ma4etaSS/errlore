"""Tests for the warning tier (errlore.consistency).

Pins the one-sided semantics from CONSISTENCY_SIGNAL_2026-07-11.md: an
inconsistent verdict warns; a consistent verdict never claims verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from errlore import AgentMemory, check_consistency
from errlore.consistency import (
    FALSE_ALARM,
    PRECISION,
    RECALL,
    RESIDUAL_AFTER_PASS,
    ConsistencyResult,
)


class TestCheckConsistency:
    def test_identical_outputs_are_stable_no_warning(self) -> None:
        r = check_consistency(["Answer: 42", "Answer: 42"])
        assert r.stable is True
        assert r.warning is None
        assert r.distinct == 1
        assert r.agreement == 1.0
        # A stable result must still disclaim verification.
        assert "NOT verification" in r.note

    def test_disagreeing_final_lines_flag_unstable(self) -> None:
        r = check_consistency(["... so the answer is 42", "... so the answer is 7"])
        assert r.stable is False
        assert r.warning is not None
        assert "likely wrong" in r.warning
        assert r.distinct == 2

    def test_final_line_mode_ignores_reasoning_differences(self) -> None:
        # Same final answer, different reasoning preamble -> stable.
        a = "First I considered X.\nThen Y.\nAnswer: yes"
        b = "A totally different chain of thought.\nAnswer: yes"
        r = check_consistency([a, b], mode="final_line")
        assert r.stable is True

    def test_full_mode_compares_whole_output(self) -> None:
        a = "reasoning one\nAnswer: yes"
        b = "reasoning two\nAnswer: yes"
        r = check_consistency([a, b], mode="full")
        assert r.stable is False  # whole text differs

    def test_majority_and_agreement_on_three_runs(self) -> None:
        r = check_consistency(["x", "x", "y"])
        assert r.stable is False
        assert r.majority == "x"
        assert r.agreement == pytest.approx(2 / 3)
        assert r.distinct == 2

    def test_similarity_loosens_matching(self) -> None:
        a = "the total revenue is forty two dollars"
        b = "the total revenue is forty two dollars exactly"
        strict = check_consistency([a, b], similarity=1.0)
        loose = check_consistency([a, b], similarity=0.7)
        assert strict.stable is False
        assert loose.stable is True

    def test_requires_two_outputs(self) -> None:
        with pytest.raises(ValueError):
            check_consistency(["only one"])

    def test_rejects_bad_mode_and_similarity(self) -> None:
        with pytest.raises(ValueError):
            check_consistency(["a", "b"], mode="nonsense")
        with pytest.raises(ValueError):
            check_consistency(["a", "b"], similarity=0.0)
        with pytest.raises(ValueError):
            check_consistency(["a", "b"], similarity=1.5)

    def test_operating_profile_matches_committed_doc(self) -> None:
        assert PRECISION == 0.86
        assert RECALL == 0.19
        assert FALSE_ALARM == 0.06
        assert RESIDUAL_AFTER_PASS == 0.61

    def test_result_is_typed(self) -> None:
        r = check_consistency(["a", "a"])
        assert isinstance(r, ConsistencyResult)


class TestFacadeIntegration:
    def test_unstable_logs_error_when_model_given(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        before = mem.stats()["errors_total"]
        r = mem.check_consistency(
            ["answer: A", "answer: B"], model="m", task_type="qa"
        )
        assert r.stable is False
        assert mem.stats()["errors_total"] == before + 1

    def test_stable_does_not_log_error(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        r = mem.check_consistency(["answer: A", "answer: A"], model="m")
        assert r.stable is True
        assert mem.stats()["errors_total"] == 0

    def test_no_model_never_logs(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        mem.check_consistency(["answer: A", "answer: B"])
        assert mem.stats()["errors_total"] == 0
