"""Tests for the Beta-Binomial harm gate (errlore.lessons.graduation).

The decision rule and its calibration are specified in
docs/SHADOW_MODE_SPEC.md; these tests pin the numbers so the gate can never
silently drift away from the validated thresholds.
"""

from __future__ import annotations

import math

import pytest

from errlore.lessons.graduation import (
    HARM_MAX,
    QUARANTINE_CONF,
    beta_sf,
    harm_probability,
    is_quarantined,
    reg_incomplete_beta,
)


def _beta_sf_numeric(x: float, a: float, b: float, steps: int = 200_000) -> float:
    """Independent survival Pr(p > x) via Simpson integration of the pdf."""
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    h = (1.0 - x) / steps
    total = 0.0
    for i in range(steps + 1):
        t = x + i * h
        if t <= 0.0 or t >= 1.0:
            continue
        weight = 1.0 if i in (0, steps) else (2.0 if i % 2 == 0 else 4.0)
        total += weight * math.exp(
            (a - 1.0) * math.log(t) + (b - 1.0) * math.log(1.0 - t) - log_beta
        )
    return total * h / 3.0


class TestBetaMath:
    def test_cdf_endpoints(self) -> None:
        assert reg_incomplete_beta(0.0, 5, 5) == 0.0
        assert reg_incomplete_beta(1.0, 5, 5) == 1.0

    def test_cdf_symmetry(self) -> None:
        # I_x(a, b) + I_{1-x}(b, a) == 1
        for x, a, b in [(0.3, 4, 7), (0.05, 7, 53), (0.7, 20, 3)]:
            left = reg_incomplete_beta(x, a, b)
            right = reg_incomplete_beta(1.0 - x, b, a)
            assert left + right == pytest.approx(1.0, abs=1e-9)

    def test_survival_matches_numeric_integration(self) -> None:
        for a, b in [(7, 53), (6, 54), (2, 38), (12, 12)]:
            cf = beta_sf(0.05, a, b)
            num = _beta_sf_numeric(0.05, a, b)
            assert cf == pytest.approx(num, abs=1e-4)


class TestSpecCalibration:
    """The exact anchor points quoted in SHADOW_MODE_SPEC.md."""

    def test_five_harms_in_twenty_quarantines(self) -> None:
        # Beta(2+5, 38+15) = Beta(7, 53); spec says Pr(p>0.05) ~= 0.973.
        assert beta_sf(HARM_MAX, 7, 53) == pytest.approx(0.973, abs=0.002)
        assert is_quarantined(success_count=15, failure_count=5) is True

    def test_four_harms_in_twenty_holds(self) -> None:
        # Beta(6, 54); spec says 0.926 -> below the 0.95 bar -> HOLD.
        assert beta_sf(HARM_MAX, 6, 54) == pytest.approx(0.926, abs=0.002)
        assert is_quarantined(success_count=16, failure_count=4) is False


class TestQuarantinePolicy:
    def test_fresh_lesson_is_not_quarantined(self) -> None:
        # A brand-new lesson (no outcomes yet) must still inject.
        assert is_quarantined(0, 0) is False

    def test_healthy_lesson_is_never_starved(self) -> None:
        # Many successes, no failures -> harm mass collapses to ~0.
        assert is_quarantined(1000, 0) is False
        assert is_quarantined(100, 0) is False

    def test_pure_failures_quarantine_fast(self) -> None:
        # Five straight failures, no successes -> clearly harmful.
        assert is_quarantined(0, 5) is True

    def test_harm_probability_monotonic_in_failures(self) -> None:
        base = harm_probability(10, 0)
        worse = harm_probability(10, 5)
        worst = harm_probability(10, 10)
        assert base < worse < worst

    def test_harm_probability_monotonic_in_successes(self) -> None:
        # More clean successes -> lower harm probability.
        assert harm_probability(50, 3) < harm_probability(5, 3)

    def test_thresholds_are_the_documented_values(self) -> None:
        assert HARM_MAX == 0.05
        assert QUARANTINE_CONF == 0.95
