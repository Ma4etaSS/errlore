# errlore + Claude Code

Give your coding agent a memory of its own failures across sessions:

- **PostToolUse hook** — every failed Bash command is logged into errlore.
  Resolve the ones you fixed (`mem.resolve(err_id, ..., lesson=...)`) or use
  `mem.add_lesson()` to capture takeaways directly.
- **SessionStart hook** — each new session begins with a briefing block of
  relevant lessons and per-tool KNOWN ISSUES, printed into the context.

## Setup (one command)

```bash
pip install errlore
errlore init claude-code            # global (~/.claude/settings.json)
errlore init claude-code --project  # or this repo only (./.claude/settings.json)
```

That writes the two hook scripts (to `~/.errlore/hooks/`) and merges them into
your `settings.json` — idempotently, preserving any hooks you already have.
Restart Claude Code (or open a new session) to pick them up. Options:
`--data-dir` (where the memory lives, default `~/.errlore/claude-code`) and
`--hooks-dir`.

Handy afterwards: `errlore stats` and `errlore lessons` to see what it learned.

### Manual setup (if you'd rather wire it yourself)

The scripts in this folder (`errlore_posttooluse.py`, `errlore_sessionstart.py`)
plus `settings.json.example` show the shape: copy them somewhere stable, fix the
paths, and merge into `.claude/settings.json`. `export ERRLORE_DATA=...` picks
the memory dir.

Notes: hook event field names can differ between Claude Code versions —
the PostToolUse script reads them defensively and never breaks the agent
loop (exit 0 on anything unexpected). Check `claude --help` / the hooks
docs for your version if events don't arrive.
