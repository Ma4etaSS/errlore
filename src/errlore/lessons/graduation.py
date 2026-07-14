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

# Decision thresholds (validated numerically in the spec).
HARM_MAX = 0.05  # h_max: a lesson is "harmful" if its harm rate exceeds this.
QUARANTINE_CONF = 0.95  # quarantine when Pr(p_h > HARM_MAX) exceeds this.


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
