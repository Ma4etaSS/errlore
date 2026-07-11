# Shadow mode — 0.3.x design spec (accepted, build after launch)

*Status: DESIGN ACCEPTED 2026-07-11. Do not implement before the public
launch. Origin: external analysis (P1 counterfactual proposal) + our
recomputed interference data + numeric validation of the decision rule.*

## Problem

On mature production systems failures are ~4% of traffic, so the classic
`report_outcome` reinforcement loop starves. Meanwhile our benchmark raw data
shows **lesson injection breaks 12–15% of previously-passing tasks**
(5/34 Haiku, 5/40 gemma; 0/0 on the realistic-convention families) while
fixing 74–100% of failing ones. See
`benchmarks/results/REPRODUCIBILITY_2026-07-11.md`.

Conclusion: on low-failure systems the abundant honest signal is not "did the
lesson fix a failure" (rare) but **"did the lesson NOT break a success"**
(every success is a trial). Shadow mode harvests exactly that.

## Mechanism

1. `inject_for(..., mode="shadow")` returns the lesson block but the caller
   does NOT put it in the main request. Main output is untouched — zero UX
   risk.
2. The pair (baseline_prompt, injected_prompt) is enqueued
   (`enqueue_counterfactual`, JSONL queue).
3. An offline worker re-runs both prompts against the same model and scores
   both with the surface's **deterministic validator** (schema validity,
   sentinel checks, exit codes, tests — never an LLM judge).
4. `report_counterfactual_outcome(...)` updates two per-lesson Beta
   posteriors:
   - harm:  p_h ~ Beta(α_h=2, β_h=38)  (prior mean 5%, biased safe)
   - fix:   p_f ~ Beta(α_f=1, β_f=9)   (prior mean 10%, weak)
   Success trial (baseline passed): B broke → α_h+=1 else β_h+=1.
   Failure trial (baseline failed): B fixed → α_f+=1 else β_f+=1.

## Graduation decision rule (validated numerically)

With h_max=0.05, f_min=0.10:

- **QUARANTINE** when Pr(p_h > h_max) > 0.95
  (e.g. 5 harms in 20 trials → 0.973; 4/20 → 0.926 stays HOLD)
- **PROMOTE** when Pr(p_h ≤ h_max) > 0.95 AND Pr(p_f > f_min) > 0.50
  (safety side needs ≈60 clean success-trials: 0-harm posteriors
  n=40 → 0.910, n=60 → 0.961, n=100 → 0.993;
  usefulness side crosses with a SINGLE observed fix: 1/1 → 0.736,
  1/5 → 0.585; zero fixes stays below: 0/0 → 0.387, 0/5 → 0.229)
- **HOLD** otherwise.

Promotion = the lesson graduates into the permanent surface (system prompt /
conventions doc / a PR) with its evidence trail (counts + replayable JSONL).
errlore is thereby also a **graduation conveyor**, not only a memory.

Properties: sequential, incremental (4 counters per lesson), reproducible by
replaying the log, asymmetric by design (strict on harm where signal is
abundant, lenient on usefulness where signal is rare).

## Selective injection (simplified from the proposal)

The 12–15% harm concentrates where lessons are irrelevant to the task. We do
NOT adopt the tuned logistic-regression scorer (tuning surface + overfitting
risk violates the zero-dep, reproducible ethos). Policy, in order:

1. **Strict task-type match** — a lesson only ever injects into its own
   task_type (already largely true; make it a hard gate in shadow mode).
2. **Retrieval floor** — below a word-overlap/embedding threshold, do not
   inject even in shadow.
3. **Per-lesson harm gate with recency blend** — effective p̂_h =
   0.7 · p̂_h(last 100 trials) + 0.3 · p̂_h(all); a lesson whose recent
   harm spikes is suppressed but can recover (no permanent starvation).

## Scope boundary (adopted verbatim as product policy)

For validator-less tasks (reports, refactors, support answers) there is **no
honest non-LLM-judge signal**; self-consistency measures stability not
correctness, user edits are sparse and confounded. Therefore:

> Shadow verification operates ONLY on validator-equipped surfaces, and we
> say so publicly. Elsewhere lessons are documentation, not verified memory,
> and their interference is explicitly unmeasured.

"We only auto-verify where verification is possible" ships as a feature.

## Production pilot (first target)

Landing-page-audit SaaS mini-audit path: validators exist (JSON schema
validity, sentinel score ≠ -1, grounding verifier). Traffic ~10/day →
safety-clearance for a lesson in ≈6 days, quarantine of a bad lesson in days.
Cost: one extra model call per sampled shadow trial (sub-cent at mini scale).

## Benchmark plan (before any claims)

Harness: extend `bench_error_reduction.py` families with irrelevant-lesson
pairings. Metrics: harm rate on previously-passing tasks (target ≤5% from
12–15% baseline via the selective policy) WITHOUT materially reducing the
74–100% fix rate; plus decision-rule trace tests (promote/hold/quarantine
transitions replayed from synthetic logs). Deterministic validators, raw
outputs committed, McNemar where paired.

## How this dies (named failure modes)

- Validator coverage in real deployments proves too thin → the pilot shows
  near-zero enqueue volume → scope shrinks to CI/test-equipped agents only.
- Shadow cost objection at scale → mitigate with sampling; if sampling makes
  decisions take months on low-traffic systems, the gate is honest but slow —
  publish the math, let users pick thresholds.
- Model drift invalidates old trials → tag trials with model id; posterior
  resets on model change (already the plan; adds cold-start latency).
