# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
