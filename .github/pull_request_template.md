## What & why

<!-- One focused change. Link the issue if any. -->

## Checklist

- [ ] `ruff check .` + `mypy` + `pytest --cov=errlore --cov-fail-under=80` pass locally
- [ ] Tests added/updated for behavior changes
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] If a claimed number changes: benchmark artifacts updated in `benchmarks/results/`
- [ ] No telemetry / no network calls added to the core
