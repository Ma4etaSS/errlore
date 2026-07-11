# Self-consistency as a quality signal: an honest one-sided result

*2026-07-11. Empirical test of the "no honest validator-less signal" claim,
using data we already had: the 2026-07-06 and 2026-07-11 default-seed runs
share identical prompts (same RNG seed), giving 96 paired independent
generations of the plain arm, five days apart, with ground-truth labels from
the deterministic validators. Pure recomputation of committed raw outputs —
no new model calls.*

**Setup.** Validator-free feature: does the model give the SAME final answer
to the identical prompt in two independent runs (exact final-line match)?
Ground truth (used only for evaluation): the validator verdict.

|                | correct | wrong |
|----------------|---------|-------|
| consistent     | 32      | **50** |
| inconsistent   | 2       | 12    |

**Findings.**

1. **"Inconsistent ⟹ wrong" is an honest negative signal: 86% precision**
   (12/14). Instability almost always marks a wrong answer.
2. **"Consistent ⟹ correct" is nearly worthless: 39% precision** (32/82).
   Stability tells you almost nothing about correctness.
3. **52% of tasks (50/96) are systematically confidently-wrong**: the model
   reproduces the SAME wrong answer across independent runs five days apart.
   This is the indistinguishability premise of the impossibility argument
   made concrete — for half this grid, correct and wrong outputs look
   identical to any consistency-based observer.

**The refined theorem (our addition to the external analysis).** The
impossibility of validator-less quality signal applies to *certifying
correctness*, not to *detecting (some) incorrectness*:

> Model error decomposes into a stochastic component and a systematic
> component. Re-run consistency is an honest detector of the stochastic
> component only (here: 86%-precision wrongness flags), and is provably blind
> to the systematic component (here: 52% of the mass), where
> confidently-wrong outputs are as stable as correct ones.

**Product consequence.** On validator-less surfaces, shadow-style machinery
can honestly ship a *warning tier* — "this output is unstable across re-runs,
likely wrong" — at the cost of one sampled re-run. It can never ship a
*verified tier* there. Verification stays scoped to validator-equipped
surfaces (docs/SHADOW_MODE_SPEC.md); instability-flagging extends honestly
beyond them.

**Caveats.** This grid is failure-rich by construction, so 52% systematic-
wrong is not a production rate; the transferable result is the asymmetry
(86% vs 39%), not the base rates. Consistency here is strict final-answer
match; looser similarity metrics would shift the numbers but not the
one-sidedness. Free-form prose tasks (no extractable final answer) need a
claim-level equivalent, which reintroduces extraction machinery — untested.
