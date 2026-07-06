#!/usr/bin/env python3
"""Claude Code SessionStart hook: brief the new session on past pitfalls.

Prints the errlore lessons + KNOWN ISSUES block to stdout; Claude Code adds
hook stdout to the session context. Empty memory -> prints nothing.
"""

import os
import sys

from errlore import AgentMemory

DATA_DIR = os.environ.get("ERRLORE_DATA", os.path.expanduser("~/.errlore/claude-code"))


def main() -> int:
    mem = AgentMemory(DATA_DIR)
    inj = mem.inject_for("starting a coding session in this workspace",
                         model="claude-code", task_type="bash")
    if inj.text:
        print(inj.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
