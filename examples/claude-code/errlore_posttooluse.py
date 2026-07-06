#!/usr/bin/env python3
"""Claude Code PostToolUse hook: failed Bash commands become errlore errors.

Reads the hook event JSON from stdin (defensively -- field names may vary
across Claude Code versions), logs failures into a shared errlore memory.
Pair with errlore_sessionstart.py, which briefs the next session on them.
"""

import json
import os
import sys

from errlore import AgentMemory

DATA_DIR = os.environ.get("ERRLORE_DATA", os.path.expanduser("~/.errlore/claude-code"))


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0  # never break the agent loop

    tool = event.get("tool_name", "")
    if tool != "Bash":
        return 0

    resp = event.get("tool_response") or {}
    if isinstance(resp, str):
        resp = {"output": resp}
    exit_code = resp.get("exit_code", resp.get("exitCode", 0))
    is_error = bool(resp.get("is_error") or resp.get("isError"))
    if not is_error and (exit_code in (0, None)):
        return 0

    command = (event.get("tool_input") or {}).get("command", "")[:160]
    output = str(resp.get("output") or resp.get("stderr") or "")[:200]
    mem = AgentMemory(DATA_DIR)
    mem.log_error("claude-code", "bash", f"CommandFailed: {command} :: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
