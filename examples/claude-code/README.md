# errlore + Claude Code

Give your coding agent a memory of its own failures across sessions:

- **PostToolUse hook** — every failed Bash command is logged into errlore.
  Resolve the ones you fixed (`mem.resolve(err_id, ..., lesson=...)`) or use
  `mem.add_lesson()` to capture takeaways directly.
- **SessionStart hook** — each new session begins with a briefing block of
  relevant lessons and per-tool KNOWN ISSUES, printed into the context.

## Setup

1. `pip install errlore`
2. Copy both scripts somewhere stable, adjust paths in
   `settings.json.example`, and merge it into your `.claude/settings.json`
   (project) or `~/.claude/settings.json` (global).
3. Optional: `export ERRLORE_DATA=...` to choose where the memory lives
   (defaults to `~/.errlore/claude-code`).

Notes: hook event field names can differ between Claude Code versions —
the PostToolUse script reads them defensively and never breaks the agent
loop (exit 0 on anything unexpected). Check `claude --help` / the hooks
docs for your version if events don't arrive.
