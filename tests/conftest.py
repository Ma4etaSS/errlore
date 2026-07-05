"""Shared fixtures. All tests run against tmp_path — no real data dirs, no server."""

from pathlib import Path

import pytest


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """Isolated data directory for a single test."""
    d = tmp_path / "agent_memory"
    d.mkdir()
    return d
