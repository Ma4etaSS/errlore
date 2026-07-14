# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-14

### Added
- **Shadow mode: counterfactual graduation.** The full mechanism from
  `docs/SHADOW_MODE_SPEC.md`. `inject_for(..., mode="shadow")` builds a lesson
  block for a parallel run that never touches the user's output (and re-includes
  quarantined lessons — the recovery route a suppressed lesson otherwise lacks).
  `enqueue_counterfactual()` durably queues the (baseline, injected) pair;
  your worker re-runs both, scores each with a deterministic validator, and
  calls `report_counterfactual_outcome(cf_id, baseline_passed, injected_passed)`.
  Two per-lesson Beta posteriors (harm + fix) drive a `graduation_status()` of
  `promote` / `hold` / `quarantine` via the validated two-gate rule (strict on
  harm, lenient on usefulness). `graduated_lessons()` surfaces lessons ready to
  bake into a permanent surface with their evidence counts. Every spec anchor
  (quarantine 5/20; promote after ~60 clean trials + 1 fix; fix/harm-clear
  posteriors 0.387/0.736/0.910/0.961/0.993) is pinned by a unit test. errlore
  never calls the model/validator — that stays the worker's job. Zero new deps.
- **Warning tier: self-consistency as an honest wrong-answer signal.** New
  `errlore.consistency` (`check_consistency` + `AgentMemory.check_consistency`):
  on validator-less surfaces, feed 2+ independent runs of the same prompt and
  errlore flags disagreement as "unstable — likely wrong" at ~86% precision
  (`benchmarks/results/CONSISTENCY_SIGNAL_2026-07-11.md`). Deliberately
  one-sided: a *stable* result is never presented as verification (61% residual
  wrongness on our grid). `final_line`/`full` modes, optional `similarity`
  loosening, and — when a model is named — an unstable verdict is logged as a
  tracked error. errlore never calls the model (offline ethos: the caller
  supplies outputs). Zero new dependencies.
- **Harm gate: interference-guarded lesson injection.** Lessons now track
  `success_count`/`failure_count` separately (the old single confidence scalar
  erased this signal — a lesson that helped 3× and hurt 3× looked untouched).
  A Beta-Binomial gate (`errlore.lessons.graduation`) withholds a lesson from
  injection once its live failure history clears a 95% credible bar that its
  harm rate exceeds 5% — calibrated to the numbers validated in
  `docs/SHADOW_MODE_SPEC.md` (5 harms/20 trials → quarantine, 4/20 → hold).
  This grounds the *harm* half of shadow mode in the live `report_outcome`
  loop and targets the measured 12–15% interference
  (`benchmarks/results/REPRODUCIBILITY_2026-07-11.md`). On by default
  (`AgentMemory(..., harm_gate=True)`); a fresh or consistently-helpful lesson
  is never gated, so good lessons are not starved. The gate is self-limiting
  (caps damage at ~4–5 harmful injections, then freezes the lesson);
  deliberate recovery/re-evaluation is deferred to shadow mode. New API:
  `AgentMemory.quarantined_lessons()` and a `lessons_quarantined` key in
  `stats()`. Zero new dependencies (regularized incomplete beta via a Lentz
  continued fraction, stdlib only).

## [0.2.2] - 2026-07-11

### Fixed
- `__version__` (and `errlore --version`) now reads the installed package
  metadata instead of a hardcoded string — 0.2.0/0.2.1 wheels reported
  themselves as 0.1.4.

## [0.2.1] - 2026-07-11

### Fixed
- **Claude Code integration: failed Bash commands are now actually captured.**
  Current Claude Code routes tool failures to the dedicated
  `PostToolUseFailure` event (`PostToolUse` fires only on success), so the
  previous hook never saw a failure — the loop was silently inert. New
  `post_tool_use_failure()` handler parses the real payload (top-level `error`
  string, `is_interrupt` guard); `errlore init claude-code` now installs and
  registers all three hooks. The legacy PostToolUse handler stays for older
  Claude Code versions. Verified against the official hooks reference; example
  shims and `settings.json.example` updated to match.
- `errlore stats` / `errlore lessons` now print the resolved data dir (to
  stderr) — running them next to a quickstart `./agent_memory` while the
  default points at `~/.errlore/claude-code` silently showed zeros.

### Changed
- Docs/site model placeholder refreshed `gpt-5.5` → `gpt-5.6` (released
  2026-07-09); site JSON-LD `softwareVersion` unstuck from 0.1.4.

## [0.2.0] - 2026-07-11 — "The Proof Release"

The headline benchmark is now proven on all three generality axes:
**seed-robustness** (5 independent runs), **task-generality** (fresh realistic
convention families), and **model-diversity** (Anthropic + Gemma). Development
status bumped Alpha → Beta.

### Added
- **Reproducibility evidence for the error-reduction benchmark**
  (`benchmarks/results/REPRODUCIBILITY_2026-07-11.md`): the headline A/B is now
  reproduced across 5 independent runs on claude-haiku-4-5 (two default-seed
  runs 5 days apart + three fresh RNG seeds). Every run: 66.7–69.8% repeat-error
  reduction, exact McNemar p between 8.4e-12 and 1.8e-9, knowledge-gap reduction
  95–100%, capability-gap −12% to 0%. Resolves the prior "single run" caveat in
  the README.
- `bench_error_reduction.py` reads `BENCH_RNG_SEED` from the environment so the
  same task families can be re-drawn with different instances for seed-robustness
  checks. Also `BENCH_FAMILIES` / `BENCH_SEED_N` / `BENCH_TEST_N` for narrowing a
  run (e.g. a small cross-family probe on a tight free-tier quota).
- **Task-generality: two new realistic knowledge-gap families** (`status_code` —
  an arbitrary internal status enum; `branch_name` — a non-standard git branch
  convention). On claude-haiku-4-5 both go 100% → 0% (24/24 → 0/24, McNemar
  p=1.19e-07), confirming the store-and-inject effect isn't specific to the
  original toy families. Report + raw outputs in `benchmarks/results/`.
- **Model-diversity: full grid on gemma-4-31b** (Cerebras — a different model
  family): 66.7% → 20.0% fail (70.0% reduction, McNemar p=2.6e-13),
  knowledge-gap 83%, capability-gap −20%; branch_name and status_code flip
  100%→0% here too. One honest wrinkle documented: `csv_order` does not
  transfer to gemma (12→12) while it flips to 0 on Haiku — lesson-following is
  itself model-dependent at the margin. `BENCH_MODEL` env override added for
  running any backend against a specific model.

## [0.1.4] - 2026-07-09

### Added
- **CLI** (`errlore` console command): `errlore init claude-code` writes the
  two Claude Code hook scripts and idempotently merges them into your
  `settings.json` (global or `--project`), preserving existing hooks — a
  one-command install instead of copy/edit/merge. Plus `errlore stats` and
  `errlore lessons`.
- `errlore.integrations.claude_code` — the hook logic (`post_tool_use`,
  `session_start`) now ships in the package and is tested, so the generated
  hooks are 3-line shims.

### Changed
- README: honest A/B framing (the knowledge-gap baseline fails by
  construction; the result shows the capture-and-re-supply loop works, not
  that memory teaches skills; single-run-at-temp-0 caveat). Coding-agent-first
  hero. Security section reworded — the sanitizer is a noise filter on the
  pattern, not an injection defense. Added a "Scale & limits" section
  (unbounded injections journal, single-process trust/vector index).
- Examples no longer hard-code aging frontier model ids; they use
  `os.getenv(...)` with a small default.

## [0.1.3] - 2026-07-06

### Fixed
- Read-cache poisoning race in `JSONLWriter.read_all`: an append landing
  mid-parse cached pre-append records under the post-append mtime/size, so
  later reads returned stale data and `atomic_update` could silently drop
  the concurrent record. Now caches only when the file is unchanged across
  the parse (pre/post stat match). Found by pre-launch adversarial audit;
  regression test added (Bug 3).
- Removed dead `JSONLWriter._invalidate_cache`.

### Changed
- README: added the error-reduction A/B benchmark section (96 paired tasks,
  63 -> 20 failures, McNemar p=1.8e-09; knowledge-gap 46/48 -> 0/48,
  capability-gap honestly worse at 17/48 -> 20/48), so the PyPI page shows
  the headline evidence, not just the retrieval table.

## [0.1.2] - 2026-07-06

### Added
- Error-reduction A/B benchmark with committed raw outputs: repeated
  workspace-convention errors 46/48 -> 0/48 (100% reduction, McNemar
  p=1.8e-09); capability-gap errors honestly unaffected.
- Claude Code hooks example (examples/claude-code/): PostToolUse failure
  logging + SessionStart lessons briefing.
- 15-second terminal demo GIF (real library output).

### Fixed
- `errlore.__version__` dunder reported 0.1.0 in the 0.1.1 wheel.

## [0.1.1] - 2026-07-06

### Fixed
- README benchmark table showed a stale MRR (0.290); the reproducible value
  is 0.488 (`python benchmarks/bench_retrieval.py`).

### Changed
- Python floor lowered to 3.10 (was 3.12): replaced `datetime.UTC` with
  `timezone.utc`; CI now tests 3.10-3.13. This unblocks Open WebUI's
  official image (Python 3.11).
- Added SECURITY.md and richer PyPI project URLs.

## [0.1.0] - 2026-07-05

### Added

- `AgentMemory` facade with four core methods: `log_error`, `resolve`, `inject_for`, `report_outcome`.
- Lesson store with JSONL persistence, deduplication (exact + 85% word overlap), confidence tracking, and automatic decay of unused lessons.
- Per-model, per-task-type error tracking and known-issue warning injection.
- `TrustEngine` with Bayesian cold-start blending, adaptive learning rate, volatility damping, domain-specific EMA bias, entropy enforcement, and temporal decay.
- `best_model(domain)` convenience method for model routing based on trust weights.
- Closed reinforcement loop: `report_outcome` reinforces/decays lessons and updates trust weights; idempotent (no double-reinforcement).
- Injection handle persistence (`injections.jsonl`) for cross-restart outcome reporting.
- Optional semantic retrieval via FastEmbed embeddings (`pip install errlore[embeddings]`), with automatic fallback to word-overlap.
- Lesson text sanitization (rejects raw JSON, code-only, too-short content).
- Atomic file rewrites for lesson/error updates (no data loss on crash).
- Thread-safe API (all public methods are safe to call from multiple threads).
- Integration examples for OpenAI, Anthropic, and LangChain (all runnable offline).
- Retrieval benchmark (`benchmarks/bench_retrieval.py`) with adversarial gold set.
- 155 tests, mypy strict, ruff linting.
