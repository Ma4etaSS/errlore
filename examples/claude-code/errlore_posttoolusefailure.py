#!/usr/bin/env python3
"""Claude Code PostToolUseFailure hook: failed Bash commands become errlore errors.

Current Claude Code routes tool FAILURES to this dedicated event (PostToolUse
fires only on success). The payload carries `tool_name`, `tool_input` and a
top-level `error` string — no `tool_response`, no structured exit code.
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

    if event.get("tool_name", "") != "Bash":
        return 0
    if bool(event.get("is_interrupt")):
        return 0  # user pressed Esc — not a failure worth learning from

    command = str((event.get("tool_input") or {}).get("command", ""))[:160]
    error = str(event.get("error") or "")[:200]
    if not command and not error:
        return 0
    mem = AgentMemory(DATA_DIR)
    mem.log_error("claude-code", "bash", f"CommandFailed: {command} :: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
