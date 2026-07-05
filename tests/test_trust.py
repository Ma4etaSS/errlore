"""Tests for errlore.trust -- TrustEngine with monotonic-growth bug regression."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from errlore.trust import FeedbackSignal, TrustEngine
from errlore.trust.engine import enforce_entropy

# ---------------------------------------------------------------------------
# REGRESSION: monotonic growth bug
# ---------------------------------------------------------------------------


class TestMonotonicGrowthRegression:
    """The original NEXUS code had a bug where outcome (always in [0,1]) was
    added directly to the logit increment, meaning even bad models always grew.

    After the fix, outcome is centered: (outcome - 0.5) * 2, so values < 0.5
    produce a NEGATIVE signal.
    """

    def test_divergence_good_vs_bad_model(self) -> None:
        """Model A gets outcome=0.9 x50, Model B gets outcome=0.1 x50.

        A must end up significantly higher than B (>0.2 gap).
        B must be BELOW the starting weight of 0.5.
        On the old buggy code, both would converge to cap (~0.92).
        """
        engine = TrustEngine(domains=("general",))
        engine.register_model("model_a", 0.5)
        engine.register_model("model_b", 0.5)

        for _ in range(50):
            engine.update("model_a", FeedbackSignal(outcome=0.9, domain="general"))
            engine.update("model_b", FeedbackSignal(outcome=0.1, domain="general"))

        wa = engine.get_weight("model_a", "general")
        wb = engine.get_weight("model_b", "general")

        assert wa > 0.5, f"Good model should be above start, got {wa}"
        assert wb < 0.5, f"Bad model should be below start, got {wb}"
        assert wa - wb > 0.2, (
            f"Gap between good and bad model must be >0.2, got {wa - wb:.4f} "
            f"(A={wa:.4f}, B={wb:.4f})"
        )

    def test_bad_outcome_decreases_weight(self) -> None:
        """A single bad outcome (0.1) must decrease the weight."""
        engine = TrustEngine(domains=("general",))
        engine.register_model("m", 0.5)

        w_before = engine.get_weight("m", "general")
        engine.update("m", FeedbackSignal(outcome=0.1))
        w_after = engine.get_weight("m", "general")

        assert w_after < w_before, f"Bad outcome should decrease: {w_before} -> {w_after}"

    def test_neutral_outcome_minimal_change(self) -> None:
        """Outcome=0.5 should produce near-zero logit delta (only decay)."""
        engine = TrustEngine(domains=("general",), decay_rate=0.0)
        engine.register_model("m", 0.6)

        engine.update("m", FeedbackSignal(outcome=0.5))
        w = engine.get_weight("m", "general")

        # Should stay very close to 0.6.  Cold-start prior blending causes a
        # small pull towards prior mean (~0.57), so allow 0.03 tolerance.
        assert abs(w - 0.6) < 0.03, f"Neutral outcome should barely move weight: {w}"


# ---------------------------------------------------------------------------
# Cap and floor
# ---------------------------------------------------------------------------


class TestCapFloor:
    def test_weight_never_exceeds_cap(self) -> None:
        engine = TrustEngine(domains=("general",), cap=0.92, floor=0.1)
        engine.register_model("m", 0.5)

        for _ in range(200):
            engine.update("m", FeedbackSignal(outcome=1.0))

        w = engine.get_weight("m", "general")
        assert w <= 0.92, f"Weight exceeded cap: {w}"

    def test_weight_never_below_floor(self) -> None:
        engine = TrustEngine(domains=("general",), cap=0.92, floor=0.1)
        engine.register_model("m", 0.5)

        for _ in range(200):
            engine.update("m", FeedbackSignal(outcome=0.0))

        w = engine.get_weight("m", "general")
        assert w >= 0.1, f"Weight below floor: {w}"

    def test_custom_cap_floor(self) -> None:
        engine = TrustEngine(domains=("general",), cap=0.8, floor=0.2)
        engine.register_model("m", 0.5)

        for _ in range(100):
            engine.update("m", FeedbackSignal(outcome=1.0))
        assert engine.get_weight("m") <= 0.8

        engine.register_model("m2", 0.5)
        for _ in range(100):
            engine.update("m2", FeedbackSignal(outcome=0.0))
        assert engine.get_weight("m2") >= 0.2


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_cold_start_blends_with_prior(self) -> None:
        """During cold start, weight should be blended between prior and logit."""
        engine = TrustEngine(
            domains=("general",),
            cold_start_threshold=15,
        )
        engine.register_model("m", 0.5)

        # After just 1 observation, weight should still be close to prior mean
        engine.update("m", FeedbackSignal(outcome=0.9))
        w = engine.get_weight("m", "general")

        # Prior mean is ~0.5, one good obs should nudge up but not wildly
        assert 0.48 < w < 0.70, f"Cold start should moderate: {w}"

    def test_after_warmup_responds_more(self) -> None:
        """After enough observations, updates should be more responsive."""
        engine = TrustEngine(
            domains=("general",),
            cold_start_threshold=10,
            decay_rate=0.0,
        )
        engine.register_model("m", 0.5)

        # Warm up with neutral outcomes
        for _ in range(15):
            engine.update("m", FeedbackSignal(outcome=0.5))

        w_before = engine.get_weight("m", "general")
        engine.update("m", FeedbackSignal(outcome=0.95))
        w_after = engine.get_weight("m", "general")

        assert w_after > w_before, "Post-warmup should respond to good signal"


# ---------------------------------------------------------------------------
# update_from_feedback
# ---------------------------------------------------------------------------


class TestUpdateFromFeedback:
    def test_positive_increases_weight(self) -> None:
        engine = TrustEngine(domains=("general",))
        engine.register_model("m", 0.5)

        w_before = engine.get_weight("m")
        engine.update_from_feedback("m", positive=True)
        w_after = engine.get_weight("m")
        assert w_after > w_before

    def test_negative_decreases_weight(self) -> None:
        engine = TrustEngine(domains=("general",))
        engine.register_model("m", 0.5)

        w_before = engine.get_weight("m")
        engine.update_from_feedback("m", positive=False)
        w_after = engine.get_weight("m")
        assert w_after < w_before

    def test_repeated_negative_drives_to_floor(self) -> None:
        engine = TrustEngine(domains=("general",), floor=0.1)
        engine.register_model("m", 0.5)

        for _ in range(50):
            engine.update_from_feedback("m", positive=False)

        assert engine.get_weight("m") <= 0.15  # near floor


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path) -> None:
        state_file = tmp_path / "trust_state.json"
        engine = TrustEngine(
            domains=("general", "code"),
            state_path=state_file,
        )
        engine.register_model("alpha", 0.7)
        engine.register_model("beta", 0.3)

        # Push weights away from initial
        for _ in range(20):
            engine.update("alpha", FeedbackSignal(outcome=0.85, domain="code"))
            engine.update("beta", FeedbackSignal(outcome=0.2, domain="general"))

        engine.save()
        assert state_file.exists()

        # Load into fresh instance
        engine2 = TrustEngine.load(
            state_file,
            domains=("general", "code"),
        )

        # Weights must match
        for model in ("alpha", "beta"):
            for domain in ("general", "code"):
                w1 = engine.get_weight(model, domain)
                w2 = engine2.get_weight(model, domain)
                assert abs(w1 - w2) < 1e-9, (
                    f"Weight mismatch for {model}/{domain}: {w1} vs {w2}"
                )

    def test_none_state_path_no_crash(self) -> None:
        """In-memory mode (state_path=None) should not crash on save."""
        engine = TrustEngine(state_path=None)
        engine.register_model("m")
        engine.update("m", FeedbackSignal(outcome=0.8))
        engine.save()  # should be a no-op

    def test_restore_clamps_corrupted_weights(self, tmp_path: Path) -> None:
        """B10: non-numeric / out-of-range weights are handled gracefully."""
        state_file = tmp_path / "corrupt.json"
        state_file.write_text(json.dumps({
            "models": {
                "good": {"general": 0.7},
                "nan_model": {"general": "not_a_number"},
                "inf_model": {"general": float("inf")},
                "low_model": {"general": 0.01},
            },
        }), encoding="utf-8")

        engine = TrustEngine.load(state_file, cap=0.92, floor=0.1)

        # good: clamped to [floor, cap] -- 0.7 is in range
        assert engine.get_weight("good") == pytest.approx(0.7)
        # nan_model: skipped entirely (non-numeric)
        from errlore.trust.engine import DEFAULT_WEIGHT
        assert engine.get_weight("nan_model") == DEFAULT_WEIGHT
        # inf_model: skipped (inf)
        assert engine.get_weight("inf_model") == DEFAULT_WEIGHT
        # low_model: clamped to floor
        assert engine.get_weight("low_model") == pytest.approx(0.1)

    def test_persisted_file_is_valid_json(self, tmp_path: Path) -> None:
        state_file = tmp_path / "state.json"
        engine = TrustEngine(state_path=state_file, domains=("general",))
        engine.register_model("m", 0.5)
        engine.update("m", FeedbackSignal(outcome=0.7))
        engine.save()

        data = json.loads(state_file.read_text())
        assert "models" in data
        assert "m" in data["models"]


# ---------------------------------------------------------------------------
# Entropy enforcement
# ---------------------------------------------------------------------------


class TestEntropyEnforcement:
    def test_single_dominant_model_gets_regularized(self) -> None:
        """When one model dominates, entropy enforcement pulls weights apart."""
        engine = TrustEngine(
            domains=("general",),
            entropy_threshold=0.7,
        )
        engine.register_model("dominant", 0.5)
        engine.register_model("weak", 0.5)

        # Push dominant way up, weak way down
        for _ in range(40):
            engine.update("dominant", FeedbackSignal(outcome=0.95))
            engine.update("weak", FeedbackSignal(outcome=0.1))

        # get_weights applies entropy enforcement
        weights = engine.get_weights("general")
        assert weights["weak"] >= engine.floor, "Weak model below floor after entropy"
        # The gap should be less extreme than raw weights
        raw_gap = engine.get_weight("dominant") - engine.get_weight("weak")
        enforced_gap = weights["dominant"] - weights["weak"]
        assert enforced_gap <= raw_gap, "Entropy should reduce gap"

    def test_enforce_entropy_direct(self) -> None:
        """Direct test of enforce_entropy function."""
        weights = {"a": 0.95, "b": 0.05}
        enforced = enforce_entropy(weights, min_weight=0.1, entropy_threshold=0.7)
        assert enforced["b"] >= 0.1
        # a should be pulled down somewhat
        assert enforced["a"] < 0.95


# ---------------------------------------------------------------------------
# Domain independence
# ---------------------------------------------------------------------------


class TestDomainIndependence:
    def test_different_domains_evolve_independently(self) -> None:
        engine = TrustEngine(domains=("code", "research"))
        engine.register_model("m", 0.5)

        # Good at code, bad at research
        for _ in range(30):
            engine.update("m", FeedbackSignal(outcome=0.9, domain="code"))
            engine.update("m", FeedbackSignal(outcome=0.15, domain="research"))

        w_code = engine.get_weight("m", "code")
        w_research = engine.get_weight("m", "research")

        assert w_code > 0.5, f"Code weight should be high: {w_code}"
        assert w_research < 0.5, f"Research weight should be low: {w_research}"
        assert w_code - w_research > 0.15, (
            f"Domains should diverge: code={w_code:.3f}, research={w_research:.3f}"
        )

    def test_unknown_domain_falls_back_to_general(self) -> None:
        engine = TrustEngine(domains=("general", "code"))
        engine.register_model("m", 0.5)

        # Unknown domain should be treated as "general"
        engine.update("m", FeedbackSignal(outcome=0.9, domain="unknown_xyz"))
        w_general = engine.get_weight("m", "general")
        assert w_general > 0.5


# ---------------------------------------------------------------------------
# LR multipliers (replaces _is_deepseek hack)
# ---------------------------------------------------------------------------


class TestLrMultipliers:
    def test_low_multiplier_limits_change_rate(self) -> None:
        """Model with lr_multiplier < 1 changes weight slower."""
        fast_engine = TrustEngine(domains=("general",), lr_multipliers={})
        slow_engine = TrustEngine(
            domains=("general",), lr_multipliers={"slow_model": 0.5}
        )

        fast_engine.register_model("fast_model", 0.5)
        slow_engine.register_model("slow_model", 0.5)

        signal = FeedbackSignal(outcome=0.9)
        for _ in range(20):
            fast_engine.update("fast_model", signal)
            slow_engine.update("slow_model", signal)

        w_fast = fast_engine.get_weight("fast_model")
        w_slow = slow_engine.get_weight("slow_model")

        # Both should increase, but slow model less
        assert w_fast > w_slow, (
            f"Slow model should lag: fast={w_fast:.4f}, slow={w_slow:.4f}"
        )


# ---------------------------------------------------------------------------
# FeedbackSignal validation
# ---------------------------------------------------------------------------


class TestFeedbackSignalValidation:
    def test_outcome_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="outcome"):
            FeedbackSignal(outcome=1.5)
        with pytest.raises(ValueError, match="outcome"):
            FeedbackSignal(outcome=-0.1)

    def test_disagreement_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="disagreement"):
            FeedbackSignal(outcome=0.5, disagreement=1.5)

    def test_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            FeedbackSignal(outcome=0.5, weight=-1.0)

    def test_nan_weight_rejected(self) -> None:
        """B9: NaN weight is rejected."""
        import math
        with pytest.raises(ValueError, match="weight"):
            FeedbackSignal(outcome=0.5, weight=math.nan)

    def test_inf_weight_rejected(self) -> None:
        """B9: inf weight is rejected."""
        with pytest.raises(ValueError, match="weight"):
            FeedbackSignal(outcome=0.5, weight=float("inf"))


# ---------------------------------------------------------------------------
# Batch update
# ---------------------------------------------------------------------------


class TestBatchUpdate:
    def test_batch_updates_all_models(self) -> None:
        engine = TrustEngine(domains=("general",))
        engine.register_model("a", 0.5)
        engine.register_model("b", 0.5)
        engine.register_model("c", 0.5)

        signal = FeedbackSignal(outcome=0.8)
        results = engine.update_batch(["a", "b", "c"], signal)

        assert len(results) == 3
        for mid in ("a", "b", "c"):
            assert results[mid] > 0.5

    def test_batch_per_model_disagreement(self) -> None:
        engine = TrustEngine(domains=("general",))
        engine.register_model("winner", 0.5)
        engine.register_model("loser", 0.5)

        signal = FeedbackSignal(outcome=0.8)
        results = engine.update_batch(
            ["winner", "loser"],
            signal,
            per_model_disagreement={"winner": 0.3, "loser": -0.3},
        )

        assert results["winner"] > results["loser"]
