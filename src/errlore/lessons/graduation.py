"""Beta-Binomial harm gate for lesson injection (0.3 interference guard).

This grounds the *harm* half of ``docs/SHADOW_MODE_SPEC.md`` in the live
``report_outcome`` loop. The spec's rigorous harm signal comes from
counterfactual shadow trials (baseline-passed-but-injection-broke-it); here we
use the abundant *live* signal — ``report_outcome`` successes/failures on
injected tasks — as a coarse proxy.

That proxy is deliberately weaker: a live task failure is not *proven* to be
caused by the injected lesson (the task might have failed anyway). So the gate
is conservative — a lesson is withheld only once its failure rate is high
enough that the harm posterior clears a 95% credible bar. This is the
"strict on harm, where signal is abundant" asymmetry from the spec; the
lenient usefulness/promotion side needs the counterfactual fix signal and
lands with shadow mode.

Measured motivation: lesson injection breaks 12-15% of previously-passing
tasks (``benchmarks/results/REPRODUCIBILITY_2026-07-11.md``). The gate exists
to drive that harm rate down without a tuned scorer.

Self-limiting by construction: once a lesson is quarantined it is no longer
injected on the live path, so it stops accruing outcomes -- the gate caps the
damage at a handful of harmful injections (~4-5 with these priors) and then
freezes. The flip side is that live quarantine is sticky: a quarantined lesson
cannot earn its way back on the live path (it is never shown again). Deliberate
re-evaluation without UX risk is the job of shadow mode (0.3.x), which re-tests
lessons in a parallel run. Until then, quarantine is the safe, sticky default;
``AgentMemory.quarantined_lessons()`` exposes the list for inspection.

Zero-dependency: the regularized incomplete beta function is implemented with
a Lentz continued fraction (Numerical Recipes ``betai``), using only
``math.lgamma`` — no scipy/numpy.
"""

from __future__ import annotations

import math

# Harm-side priors, validated in SHADOW_MODE_SPEC.md.
# p_h ~ Beta(alpha_h, beta_h), prior mean 5%, biased toward "safe".
_ALPHA_H = 2.0
_BETA_H = 38.0

# Fix-side priors (SHADOW_MODE_SPEC.md).
# p_f ~ Beta(alpha_f, beta_f), prior mean 10%, weak.
_ALPHA_F = 1.0
_BETA_F = 9.0

# Decision thresholds (validated numerically in the spec).
HARM_MAX = 0.05  # h_max: a lesson is "harmful" if its harm rate exceeds this.
FIX_MIN = 0.10  # f_min: a lesson is "useful" if its fix rate exceeds this.
QUARANTINE_CONF = 0.95  # quarantine when Pr(p_h > HARM_MAX) exceeds this.
PROMOTE_SAFE_CONF = 0.95  # promote needs Pr(p_h <= HARM_MAX) above this...
PROMOTE_USEFUL_CONF = 0.50  # ...AND Pr(p_f > FIX_MIN) above this.


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    max_iter = 300
    eps = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def reg_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` -- the Beta(a, b) CDF at *x*.

    Args:
        x: Evaluation point in ``[0, 1]``.
        a: First shape parameter (> 0).
        b: Second shape parameter (> 0).

    Returns:
        ``P(X <= x)`` for ``X ~ Beta(a, b)``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log(1.0 - x))
    # Use the continued fraction that converges fastest for this x.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_sf(x: float, a: float, b: float) -> float:
    """Survival function ``Pr(p > x)`` for ``p ~ Beta(a, b)``."""
    return 1.0 - reg_incomplete_beta(x, a, b)


def harm_probability(success_count: int, failure_count: int) -> float:
    """``Pr(harm rate > HARM_MAX)`` under the live-proxy harm posterior.

    Posterior is ``Beta(alpha_h + failures, beta_h + successes)``: a failure on
    an injected task pushes harm mass up, a success pushes it down.

    Args:
        success_count: Injected tasks that succeeded (non-harm trials).
        failure_count: Injected tasks that failed (candidate-harm trials).

    Returns:
        Posterior probability the lesson's harm rate exceeds ``HARM_MAX``.
    """
    a = _ALPHA_H + max(0, failure_count)
    b = _BETA_H + max(0, success_count)
    return beta_sf(HARM_MAX, a, b)


def is_quarantined(success_count: int, failure_count: int) -> bool:
    """Whether a lesson should be withheld from injection.

    ``True`` when ``Pr(p_h > HARM_MAX) > QUARANTINE_CONF``. Calibration matches
    the spec: 5 harms in 20 trials quarantines (~0.973); 4 in 20 holds
    (~0.926); a fresh lesson (0, 0) and a healthy one (many successes, no
    failures) are never quarantined, so good lessons are not starved.

    Args:
        success_count: Injected tasks that succeeded.
        failure_count: Injected tasks that failed.

    Returns:
        ``True`` if the lesson is quarantined (do not inject).
    """
    return harm_probability(success_count, failure_count) > QUARANTINE_CONF


# ---------------------------------------------------------------------------
# Shadow-mode graduation (counterfactual trials -> promote/hold/quarantine)
# ---------------------------------------------------------------------------
#
# Counterfactual trials feed two per-lesson posteriors (SHADOW_MODE_SPEC.md):
#
#   Success trial (baseline PASSED):  injected broke it -> harm_break else harm_keep
#   Failure trial (baseline FAILED):  injected fixed it -> fix_yes    else fix_no
#
# harm posterior: Beta(_ALPHA_H + harm_break, _BETA_H + harm_keep)
# fix  posterior: Beta(_ALPHA_F + fix_yes,    _BETA_F + fix_no)


def harm_clear_probability(harm_break: int, harm_keep: int) -> float:
    """``Pr(p_h <= HARM_MAX)`` under the counterfactual harm posterior."""
    return reg_incomplete_beta(
        HARM_MAX, _ALPHA_H + max(0, harm_break), _BETA_H + max(0, harm_keep)
    )


def fix_useful_probability(fix_yes: int, fix_no: int) -> float:
    """``Pr(p_f > FIX_MIN)`` under the counterfactual fix posterior."""
    return beta_sf(FIX_MIN, _ALPHA_F + max(0, fix_yes), _BETA_F + max(0, fix_no))


def decide(
    harm_break: int,
    harm_keep: int,
    fix_yes: int,
    fix_no: int,
) -> str:
    """Graduation verdict for a lesson from its counterfactual trial counts.

    Implements the validated two-gate rule from SHADOW_MODE_SPEC.md:

    * ``"quarantine"`` when ``Pr(p_h > HARM_MAX) > QUARANTINE_CONF``
      (harm gate; checked first — safety dominates).
    * ``"promote"`` when ``Pr(p_h <= HARM_MAX) > PROMOTE_SAFE_CONF`` AND
      ``Pr(p_f > FIX_MIN) > PROMOTE_USEFUL_CONF`` (safe enough AND useful).
    * ``"hold"`` otherwise (insufficient evidence either way).

    The rule is asymmetric by design: strict on harm (abundant signal), lenient
    on usefulness (rare signal). Calibration anchors from the spec: quarantine
    at 5 harm-breaks / 20 success-trials; promotion needs ~60 clean
    success-trials on the safety side and a single observed fix on the
    usefulness side.

    Returns:
        One of ``"promote"``, ``"hold"``, ``"quarantine"``.
    """
    harm_break = max(0, harm_break)
    harm_keep = max(0, harm_keep)
    # One survival-function evaluation drives both gates: Pr(p_h > HARM_MAX)
    # for quarantine and its complement Pr(p_h <= HARM_MAX) for promote-safe.
    harm_exceed = beta_sf(HARM_MAX, _ALPHA_H + harm_break, _BETA_H + harm_keep)
    if harm_exceed > QUARANTINE_CONF:
        return "quarantine"
    safe = (1.0 - harm_exceed) > PROMOTE_SAFE_CONF
    useful = fix_useful_probability(fix_yes, fix_no) > PROMOTE_USEFUL_CONF
    if safe and useful:
        return "promote"
    return "hold"
