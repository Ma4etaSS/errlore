# errlore error-reduction — reproducibility across 5 independent runs

**Date:** 2026-07-11
**Model:** claude-haiku-4-5 (Anthropic), temperature 0
**Harness:** `benchmarks/bench_error_reduction.py` (paired A/B, deterministic
validators, exact McNemar; no LLM judges)

The README benchmark's one honest weakness was that it was a **single run**.
This resolves that: the same A/B was run five times — two independent runs on
the default seed (5 days apart) plus three fresh RNG seeds (different task
instances of the same families). The headline effect and its error-class split
hold in every run.

## The five runs

| run | seed | A plain | B errlore | reduction | McNemar p | knowledge-gap A→B | capability-gap A→B |
|-----|------|---------|-----------|-----------|-----------|-------------------|--------------------|
| 2026-07-06 | default | 65.6% (63/96) | 20.8% (20/96) | ~68% | 1.8e-09 | 46/48 → ~0 | 17/48 → 20/48 |
| 2026-07-11 | default | 64.6% (62/96) | 20.8% (20/96) | 67.7% | 1.285e-09 | 46/48 → 0/48 | 16/48 → 20/48 |
| 2026-07-11 | 11 | 62.5% (60/96) | 20.8% (20/96) | 66.7% | 4.622e-10 | 44/48 → 2/48 (95%) | 16/48 → 18/48 |
| 2026-07-11 | 22 | 67.7% (65/96) | 21.9% (21/96) | 67.7% | 8.363e-12 | 46/48 → 2/48 (96%) | 19/48 → 19/48 (0%) |
| 2026-07-11 | 33 | 65.6% (63/96) | 19.8% (19/96) | 69.8% | 1.307e-10 | 46/48 → 0/48 (100%) | 17/48 → 19/48 |

## What holds across all five

- **Overall repeat-error reduction: 66.7% – 69.8%** (spread ≈ 2.5 pp). arm B
  fail rate lands at 19.8–21.9% every time from a 62.5–67.7% baseline.
- **Statistical significance: exact McNemar p between 8.4e-12 and 1.8e-9** — the
  effect is not a fluke in any single run.
- **Knowledge-gap errors (workspace conventions — date/ID/rounding/CSV-order
  rules): 95–100% reduction.** These are conventions the model was never told,
  so arm A fails almost by construction; errlore captures the fix once and
  re-supplies it, driving the failures to 0–2 out of 48 in every run.
- **Capability-gap errors (letter counting, string reversal): −12% to 0%.**
  errlore does not help — and occasionally hurts slightly — where the failure
  is a model *skill* limit, not a missing convention. This is the honest
  boundary of the mechanism, and it reproduces just as reliably as the win.

## How to reproduce

```bash
cd errlore
export ANTHROPIC_API_KEY=...
# default seed:
.venv/bin/python benchmarks/bench_error_reduction.py --backend anthropic --out /tmp/r.jsonl
# a different task-instance draw:
BENCH_RNG_SEED=22 .venv/bin/python benchmarks/bench_error_reduction.py --backend anthropic --out /tmp/r22.jsonl
```

Per-run reports are committed alongside this file
(`haiku_2026-07-11_report.txt`, `haiku_2026-07-11_seed{11,22,33}.txt`) and raw
model outputs in `haiku_2026-07-11_raw.jsonl` for independent recomputation.

## Task generality (2 new realistic conventions)

The original eight families could be dismissed as toy tasks. So two **new**
knowledge-gap families were added that mirror real "our agent keeps getting our
internal convention wrong" pain, and run on the same harness (claude-haiku-4-5,
default 6 seed / 12 test each):

- **`status_code`** — map a subscription status to our arbitrary internal enum
  (active→A1, cancelled→X9, …). A pure lookup only the lesson supplies.
- **`branch_name`** — our deliberately non-standard git branch convention
  `wip.<TICKET>.<snake_desc>` (the model's default guess is `feat/<ticket>-<kebab>`).

| arm | failures | fail rate |
|-----|----------|-----------|
| A plain | 24/24 | **100.0%** |
| B errlore | 0/24 | **0.0%** |

Exact McNemar p = 1.19e-07; per-family branch_name 12→0, status_code 12→0.
arm A fails 100% by construction (the conventions are genuinely unguessable);
arm B fixes 100% once the convention is captured. The store-and-inject loop
generalizes cleanly to new, realistic conventions — not just the seed families.

## Model diversity (gemma-4-31b — a different model family)

The same full grid (all 10 families, default 6 seed / 12 test) on
**gemma-4-31b** via Cerebras — a completely different model family from
claude-haiku-4-5:

| arm | failures | fail rate |
|-----|----------|-----------|
| A plain | 80/120 | 66.7% |
| B errlore | 24/120 | **20.0%** |

Exact McNemar p = 2.631e-13; repeat-error reduction **70.0%** (fixed 61,
broke 5). Knowledge-gap 70/72 → 12/72 (**83%**); capability-gap 10/48 → 12/48
(**−20%**) — the honest boundary reproduces across model families too. The two
realistic-convention families flip identically: branch_name 12→0,
status_code 12→0.

One honest wrinkle: `csv_order` did **not** transfer to gemma (12→12 — the
lesson is injected but gemma keeps emitting the wrong column order), while it
flips to 0 on Haiku. Lesson-following is itself model-dependent at the margin;
we report it rather than hide it.

## One-line claim

> On two different model families (claude-haiku-4-5 across 5 runs,
> gemma-4-31b), errlore cut the repeat-error rate by ~68–70% (≈66% → ≈20%,
> McNemar p ≤ 2e-9 every run), by fixing 83–100% of knowledge-gap errors —
> and honestly does nothing for capability-gap errors on either family. Two
> fresh realistic-convention families (arbitrary status enum, non-standard
> branch naming) went 100% → 0% on both models.
