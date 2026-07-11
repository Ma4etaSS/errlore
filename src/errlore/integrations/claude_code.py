"""Claude Code hook logic, shipped so the generated hooks are 3-line shims.

``errlore init claude-code`` writes two tiny scripts that call the two
functions here, so the real (tested) logic lives in the package instead of in
copy-pasted example files:

* :func:`post_tool_use_failure` -- a PostToolUseFailure hook: a failed Bash
  command becomes an errlore error. Current Claude Code routes tool failures
  to this dedicated event (PostToolUse fires only on SUCCESS) with a top-level
  ``error`` string and no ``tool_response``.
* :func:`post_tool_use` -- legacy PostToolUse handler, kept for older Claude
  Code versions whose PostToolUse payload carried ``exit_code``/``is_error``.
  On current versions a failed command never reaches PostToolUse, so this
  stays a harmless no-op.
* :func:`session_start` -- a SessionStart hook: print the lessons + KNOWN ISSUES
  briefing to stdout (Claude Code adds hook stdout to the session context).

Both are defensive by contract: they read best-effort, never raise into the
agent loop, and always return ``0``. Field names differ across Claude Code
versions, so the event is parsed loosely.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from errlore import AgentMemory


def data_dir() -> str:
    """Where the Claude Code memory lives. ``ERRLORE_DATA`` overrides it."""
    return os.environ.get("ERRLORE_DATA", os.path.expanduser("~/.errlore/claude-code"))


def post_tool_use(event_json: str | None = None) -> int:
    """Log a failed Bash command as an errlore error.

    Args:
        event_json: The raw hook-event JSON. ``None`` reads ``sys.stdin``
            (the real hook path); tests pass a string.

    Returns:
        Always ``0`` -- a hook must never break the agent loop.
    """
    raw = event_json if event_json is not None else sys.stdin.read()
    try:
        event: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0
    if not isinstance(event, dict):
        return 0

    if event.get("tool_name", "") != "Bash":
        return 0

    resp = event.get("tool_response") or {}
    if isinstance(resp, str):
        resp = {"output": resp}
    if not isinstance(resp, dict):
        return 0

    exit_code = resp.get("exit_code", resp.get("exitCode", 0))
    is_error = bool(resp.get("is_error") or resp.get("isError"))
    if not is_error and exit_code in (0, None):
        return 0

    tool_input = event.get("tool_input") or {}
    command = str(tool_input.get("command", "") if isinstance(tool_input, dict) else "")[:160]
    output = str(resp.get("output") or resp.get("stderr") or "")[:200]
    try:
        AgentMemory(data_dir()).log_error(
            "claude-code", "bash", f"CommandFailed: {command} :: {output}",
        )
    except Exception:  # never break the agent loop
        return 0
    return 0


def post_tool_use_failure(event_json: str | None = None) -> int:
    """Log a failed Bash command from a PostToolUseFailure event.

    Current Claude Code fires this dedicated event when a tool call fails
    (PostToolUse only fires on success). The payload carries ``tool_name``,
    ``tool_input`` and a top-level ``error`` string — there is no
    ``tool_response`` and no structured exit code.

    Args:
        event_json: The raw hook-event JSON. ``None`` reads ``sys.stdin``
            (the real hook path); tests pass a string.

    Returns:
        Always ``0`` -- a hook must never break the agent loop.
    """
    raw = event_json if event_json is not None else sys.stdin.read()
    try:
        event: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0
    if not isinstance(event, dict):
        return 0

    if event.get("tool_name", "") != "Bash":
        return 0
    # A user hitting Esc is not a failure worth learning from.
    if bool(event.get("is_interrupt")):
        return 0

    tool_input = event.get("tool_input") or {}
    command = str(tool_input.get("command", "") if isinstance(tool_input, dict) else "")[:160]
    error = str(event.get("error") or "")[:200]
    if not command and not error:
        return 0
    try:
        AgentMemory(data_dir()).log_error(
            "claude-code", "bash", f"CommandFailed: {command} :: {error}",
        )
    except Exception:  # never break the agent loop
        return 0
    return 0


def session_start() -> int:
    """Print the lessons + KNOWN ISSUES briefing for the new session.

    Empty memory prints nothing. Returns ``0`` always.
    """
    try:
        mem = AgentMemory(data_dir())
        inj = mem.inject_for(
            "starting a coding session in this workspace",
            model="claude-code",
            task_type="bash",
        )
        if inj.text:
            print(inj.text)
    except Exception:  # never break the agent loop
        return 0
    return 0
