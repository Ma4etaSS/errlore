"""Tests for the errlore CLI and the Claude Code hook logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from errlore import AgentMemory
from errlore.cli import main
from errlore.integrations.claude_code import post_tool_use, session_start

# --------------------------------------------------------------------------
# Claude Code hook logic
# --------------------------------------------------------------------------


def test_post_tool_use_logs_failed_bash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ERRLORE_DATA", str(tmp_path))
    event = json.dumps({
        "tool_name": "Bash",
        "tool_input": {"command": "pytest -x"},
        "tool_response": {"exit_code": 1, "output": "1 failed"},
    })
    assert post_tool_use(event) == 0
    assert AgentMemory(tmp_path).stats()["errors_total"] == 1


def test_post_tool_use_ignores_success_nonbash_and_junk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ERRLORE_DATA", str(tmp_path))
    # success
    assert post_tool_use(json.dumps({"tool_name": "Bash", "tool_response": {"exit_code": 0}})) == 0
    # non-Bash
    nonbash = json.dumps({"tool_name": "Read", "tool_response": {"is_error": True}})
    assert post_tool_use(nonbash) == 0
    # not JSON / not a dict — must never raise
    assert post_tool_use("not json at all") == 0
    assert post_tool_use("[1, 2, 3]") == 0
    assert AgentMemory(tmp_path).stats()["errors_total"] == 0


def test_session_start_briefs_and_is_silent_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("ERRLORE_DATA", str(tmp_path))
    # empty memory -> prints nothing
    assert session_start() == 0
    assert capsys.readouterr().out.strip() == ""
    # with a relevant lesson -> briefing appears
    AgentMemory(tmp_path).add_lesson(
        "bash command in this workspace fails on missing venv",
        "activate .venv before running pytest",
        task_type="bash",
    )
    assert session_start() == 0
    out = capsys.readouterr().out
    assert "LESSONS FROM PAST FAILURES" in out


# --------------------------------------------------------------------------
# CLI: init claude-code
# --------------------------------------------------------------------------


def test_init_claude_code_writes_hooks_and_merges_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    hooks_dir = tmp_path / "hooks"
    data_dir = tmp_path / "data"
    rc = main([
        "init", "claude-code", "--project",
        "--data-dir", str(data_dir), "--hooks-dir", str(hooks_dir),
    ])
    assert rc == 0

    # hook shims written + executable
    post = hooks_dir / "errlore_posttooluse.py"
    session = hooks_dir / "errlore_sessionstart.py"
    assert post.exists() and session.exists()
    assert "post_tool_use" in post.read_text()
    assert str(data_dir) in post.read_text()  # data dir pinned into the shim

    # settings.json wired
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    ptu = settings["hooks"]["PostToolUse"]
    ss = settings["hooks"]["SessionStart"]
    assert any("errlore_posttooluse.py" in h["command"]
               for g in ptu for h in g["hooks"])
    assert any("errlore_sessionstart.py" in h["command"]
               for g in ss for h in g["hooks"])
    assert ptu[0]["matcher"] == "Bash"


def test_init_claude_code_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    args = ["init", "claude-code", "--project",
            "--data-dir", str(tmp_path / "d"), "--hooks-dir", str(tmp_path / "h")]
    assert main(args) == 0
    assert main(args) == 0  # run twice
    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    # no duplicated hook entries
    assert len(settings["hooks"]["PostToolUse"]) == 1
    assert len(settings["hooks"]["SessionStart"]) == 1


def test_init_preserves_existing_foreign_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    claude = tmp_path / ".claude"
    claude.mkdir()
    (claude / "settings.json").write_text(json.dumps({
        "hooks": {"PostToolUse": [{"matcher": "Edit", "hooks": [
            {"type": "command", "command": "echo mine"}]}]},
        "otherKey": 42,
    }))
    assert main(["init", "claude-code", "--project",
                 "--data-dir", str(tmp_path / "d"), "--hooks-dir", str(tmp_path / "h")]) == 0
    settings = json.loads((claude / "settings.json").read_text())
    assert settings["otherKey"] == 42  # untouched
    cmds = [h["command"] for g in settings["hooks"]["PostToolUse"] for h in g["hooks"]]
    assert "echo mine" in cmds  # foreign hook preserved
    assert any("errlore_posttooluse.py" in c for c in cmds)  # ours added


# --------------------------------------------------------------------------
# CLI: stats / lessons
# --------------------------------------------------------------------------


def test_stats_and_lessons_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    mem = AgentMemory(tmp_path)
    mem.add_lesson("thing breaks", "do the fix", task_type="bash")
    assert main(["stats", "--data-dir", str(tmp_path), "--json"]) == 0
    assert "lessons_total" in capsys.readouterr().out
    assert main(["lessons", "--data-dir", str(tmp_path)]) == 0
    assert "do the fix" in capsys.readouterr().out
