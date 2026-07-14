"""Warning tier: self-consistency as an honest one-sided quality signal.

For validator-less surfaces (prose, judgement, answers with no deterministic
oracle) errlore cannot *verify* correctness. But re-run consistency is an
honest detector of the *stochastic* component of error: run the same prompt
twice (temperature > 0) and if the final answers disagree, at least one run is
wrong. Our labeled benchmark puts this "inconsistent -> wrong" flag at **86%
precision** (raw data: ``benchmarks/results/CONSISTENCY_SIGNAL_2026-07-11.md``).

The theorem is deliberately one-sided (see the same doc):

* Inconsistency is an honest wrongness flag (high precision).
* Consistency is NOT verification. A passed check leaves the *systematic*
  error untouched -- on our failure-rich grid, 61% of passed outputs were
  still wrong. This module states that plainly and never presents a "stable"
  result as a guarantee.

errlore never calls the model here -- consistent with the offline, no-server
ethos, the caller supplies the outputs (or runs its own sampler and passes
them in). The cost is one sampled re-run, paid by the caller.

Operating profile below is grid-specific; the transferable result is the
asymmetry (86% vs 39%), not the base rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Honest operating profile from CONSISTENCY_SIGNAL_2026-07-11.md (96 paired
# tasks, deterministic-validator ground truth). Grid-specific; exposed so
# product copy stays grounded rather than invented.
PRECISION = 0.86  # P(wrong | inconsistent) = 12/14
RECALL = 0.19  # P(inconsistent | wrong) = 12/62 -- catches only the stochastic slice
FALSE_ALARM = 0.06  # P(inconsistent | correct) = 2/34
RESIDUAL_AFTER_PASS = 0.61  # P(wrong | consistent) = 50/82 on this failure-rich grid

_UNSTABLE_WARNING = (
    "Unstable across re-runs -- likely wrong. This flag is ~86% precision on a "
    "labeled benchmark (benchmarks/results/CONSISTENCY_SIGNAL_2026-07-11.md) but "
    "catches only ~1/5 of errors, so a clean result is NOT a guarantee."
)
_STABLE_NOTE = (
    "Stable across re-runs. This is NOT verification: a passed consistency check "
    "cannot touch the systematic-error component (61% of passed outputs were still "
    "wrong on our failure-rich grid). Use only as a cheap wrong-answer detector."
)


def _final_line(text: str) -> str:
    """Last non-empty line, stripped -- the 'final answer' comparison key."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def _word_overlap(a: str, b: str) -> float:
    """Word-level overlap |A & B| / max(|A|, |B|)."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


@dataclass(slots=True)
class ConsistencyResult:
    """Verdict from :func:`check_consistency`.

    Attributes:
        stable: True when every run produced an equivalent final answer.
        n_runs: Number of outputs compared.
        distinct: Number of distinct answer clusters found.
        agreement: Fraction of runs in the largest cluster (``[0, 1]``).
        majority: The most common answer key (useful when unstable).
        warning: Honest "likely wrong" text when unstable, else ``None``.
        note: Always states that a stable result is not verification.
    """

    stable: bool
    n_runs: int
    distinct: int
    agreement: float
    majority: str
    warning: str | None
    note: str = field(default=_STABLE_NOTE)


def check_consistency(
    outputs: list[str],
    *,
    mode: str = "final_line",
    similarity: float = 1.0,
) -> ConsistencyResult:
    """Flag stochastic-component wrongness via re-run consistency.

    Args:
        outputs: Two or more model outputs for the *same* prompt, produced by
            independent runs (temperature > 0 makes the signal meaningful).
        mode: How to derive the comparison key from each output.
            ``"final_line"`` (default) compares the last non-empty line -- the
            validated setting. ``"full"`` compares the whole output stripped.
        similarity: Equivalence threshold in ``(0, 1]``. ``1.0`` (default)
            requires exact key match -- the validated strict setting. Below
            1.0, two keys count as equal when their word overlap is at least
            this value (looser matching shifts the numbers but not the
            one-sidedness).

    Returns:
        A :class:`ConsistencyResult`. When unstable, ``warning`` carries the
        honest ~86%-precision text; a stable result never claims verification.

    Raises:
        ValueError: If fewer than two outputs are supplied, or *mode* /
            *similarity* is out of range.
    """
    if len(outputs) < 2:
        raise ValueError("check_consistency needs at least two outputs")
    if mode not in ("final_line", "full"):
        raise ValueError(f"unknown mode: {mode!r}")
    if not (0.0 < similarity <= 1.0):
        raise ValueError("similarity must be in (0, 1]")

    extract = _final_line if mode == "final_line" else (lambda t: t.strip())
    keys = [extract(o) for o in outputs]
    n = len(keys)

    def _equivalent(a: str, b: str) -> bool:
        if a == b:
            return True
        if similarity < 1.0:
            return _word_overlap(a, b) >= similarity
        return False

    # Union-find clustering: two answers share a cluster when they are
    # pairwise equivalent, closed transitively. For the strict default
    # (similarity == 1.0, exact equality) equivalence is already transitive,
    # so this matches a plain set; for looser similarity it keeps `distinct`,
    # `agreement`, and `stable` independent of input order.
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if _equivalent(keys[i], keys[j]):
                parent[_find(i)] = _find(j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(_find(i), []).append(i)

    members = list(clusters.values())
    largest = max(members, key=len)
    stable = len(members) == 1
    return ConsistencyResult(
        stable=stable,
        n_runs=n,
        distinct=len(members),
        agreement=len(largest) / n,
        majority=keys[largest[0]],
        warning=None if stable else _UNSTABLE_WARNING,
    )
