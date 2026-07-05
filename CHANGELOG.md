# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
