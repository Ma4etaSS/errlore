# Contributing to errlore

Thanks for taking the time. Two rules keep this project trustworthy:

1. **Every number needs a benchmark.** Claims in the README/site must be
   backed by a committed, reproducible benchmark with raw outputs. If your
   change alters a claimed number, update the benchmark artifacts with it.
2. **Features land with their evidence.** A new capability (e.g. a retrieval
   mode, a routing policy) ships together with a benchmark demonstrating the
   effect — or it ships marked *experimental* and stays out of the headline.

## Dev setup

```bash
git clone https://github.com/Ma4etaSS/errlore
cd errlore
python -m venv .venv && . .venv/bin/activate
pip install -e .[dev]
```

## Quality gate (same as CI and the pre-push hook)

```bash
ruff check .
mypy
pytest --cov=errlore --cov-fail-under=80
```

CI runs the same on Python 3.10–3.13. Keep `mypy` strict-clean and coverage
above the bar; add tests for any behavior change.

## Pull requests

- One focused change per PR; reference the issue if there is one.
- Update `CHANGELOG.md` under `[Unreleased]` (Keep a Changelog format).
- No telemetry, no network calls in the core — errlore stays offline-first.
- Benchmarks live in `benchmarks/`; committed results in `benchmarks/results/`
  (reports as text; raw outputs as `.jsonl`, force-added past the gitignore).

## Releases (maintainers)

Tag-driven: bump `version` in `pyproject.toml`, move `[Unreleased]` to a
dated section, then `git tag vX.Y.Z && git push origin main vX.Y.Z`. CI runs
the gate, builds, publishes to PyPI, and attaches artifacts to the GitHub
release.

## Security

See [SECURITY.md](SECURITY.md) — report privately, not via public issues.
