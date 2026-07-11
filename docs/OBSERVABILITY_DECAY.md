# The observability-decay law (working notes)

*2026-07-11. Product of a three-round external-analysis exchange plus our own
measured data. Status: theory notes — the positioning backbone for a future
essay, not shipped claims. Everything empirical referenced here is committed
in benchmarks/results/.*

## The law (two phases)

Let R be system reliability (share of tasks done correctly), s the
systematic share of remaining errors (stable, reproducible mistakes — see
CONSISTENCY_SIGNAL_2026-07-11.md), and C(R) the cost of one unit of honest
quality signal.

**Phase 1 — stochastic-error regime.** Signal comes free from observing
production failures. Information per run ∝ (1−R), so

    C(R) ∝ 1 / (1 − R)

— the standard diminishing-returns curve (each next bug costs more to see).

**Phase 2 — systematic-error regime.** As lessons graduate into prompts and
code, the stochastic component gets fixed FIRST (it's the observable one),
so the residual error concentrates in the systematic share: s/(1−R) → 1.
Systematic errors are invisible to consistency/re-run signals (measured:
recall 19%, residual-after-pass 61%), so free and re-run signals both die and

    C(R) → cost of an external validator, or ∞ where none exists.

The mechanism that produces maturity (graduating lessons) is exactly the
mechanism that destroys the signal supply. This resolves our paradox from
first principles rather than as an anecdote.

## The verifier-ceiling theorem — with the hole patched

External proposal: *asymptotic system quality cannot exceed the quality of
the cheapest verifier you can afford to run routinely.*

**The hole:** as stated, it's false. A weak verifier with INDEPENDENT
(stochastic) errors can be amplified beyond its single-shot accuracy by
repetition/ensembling (Condorcet): ask it k times, majority-vote, error
falls exponentially. Cheap-but-noisy verifiers do not cap you at their raw
accuracy.

**The patch (and the stronger law):** verifier error decomposes exactly like
system error — stochastic + systematic. The stochastic part is amplifiable
away at O(k) cost; the systematic part is not, at any cost, by any number of
re-queries of the same verifier. Therefore:

> **Asymptotic quality is bounded by the SYSTEMATIC-ERROR FLOOR of the
> cheapest affordable verifier — not by its raw accuracy.**

Pleasing symmetry: the same decomposition that gave the one-sided
consistency result for generators (Q2) gives the ceiling for verifiers (Q7).

## Confirmation from known systems

The law retro-predicts the last decade of self-improvement successes:
self-play worked where the verifier is PERFECT and FREE — game rules
(AlphaZero: V=1, C_val≈0 → no ceiling, superhuman play), proof checkers
(formal math), compilers/tests (code RL). It stalls where verification is
expensive or absent — open-ended prose, judgment, taste. LLM-agent work sits
mostly on the wrong side of that line, which is why "agent reliability"
feels stuck while board games fell.

## Predictions (testable on our own systems)

1. Cost-per-signal-bit curves upward ≈ 1/(1−R) while errors are stochastic,
   then inflects when s/(1−R) crosses ~½ (we lack historical cost
   bookkeeping to fit this today; start recording it).
2. s/(1−R) grows monotonically with system age — random errors get fixed
   first. (Our single measurement: s-dominated already at maturity — 52% of
   grid tasks systematic-wrong; longitudinal confirmation pending.)
3. Teams adopt paid signal (canaries, shadow runs) when free-failure rate
   drops below the point where a decision-worth of evidence takes longer
   than the decision deadline. (Matches us: shadow-mode design began at ~4%
   production failure rate — a 10-failure evidence batch would take ~25 days
   at our volume.)
4. A system's quality plateaus at its cheapest verifier's systematic floor.
   (Matches: our audit product's quality is capped by what schema+grounding
   validators can see, not by the generator model.)

## Strategic consequence

If the ceiling is the verifier's systematic floor, then the durable value in
the agent stack is **verifier-building**: turning unverifiable surfaces into
validator-equipped ones (schemas, invariants, claim-checks, replayable
oracles). Memory (errlore today) is the conveyor that exploits verifiers;
the scarce asset it depends on — and the long-term product direction — is
manufacturing them. A lesson-graduation pipeline and a validator-authoring
toolkit are two halves of the same machine.

## Honest status

The "law" fits one team's data and known self-play history; it is a strong
hypothesis with named predictions, not established theory. Nearest prior
art, none of which states the signal-COST formulation: software reliability
growth models (defect-count decay, not signal cost), bandit exploration
costs, Goodhart dynamics, scalable-oversight / weak-to-strong limits.
