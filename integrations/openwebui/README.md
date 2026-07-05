# errlore for Open WebUI

Give your Open WebUI assistant a memory that learns from failures: lessons and
per-model KNOWN ISSUES are injected into every chat, and your feedback closes
the reinforcement loop.

Two functions work as a pair:

| File | Type | What it does |
|---|---|---|
| `errlore_memory_filter.py` | Filter | Injects relevant lessons + KNOWN ISSUES as a system message before each model call; remembers the injection handle per chat |
| `errlore_feedback_action.py` | Action | Adds a button under assistant messages: good → reinforce the injected lessons; bad → log the error and optionally capture a new lesson |

## Install

1. Open WebUI → **Admin Panel → Functions → Import / New**.
2. Paste `errlore_memory_filter.py`, save. Repeat for `errlore_feedback_action.py`.
3. The `requirements: errlore` frontmatter installs the library automatically
   from PyPI (https://pypi.org/project/errlore/).
4. Enable the Filter globally (or per-model) and the Action.

Both functions share one data directory (`data_dir` valve, default
`/app/backend/data/errlore`). All data stays on your machine — plain JSONL
files, no external calls.

## Seed it

The memory starts empty. Two ways to grow it:

- Click the feedback button on bad responses and type a takeaway — it becomes
  a lesson injected into future similar chats.
- Seed lessons programmatically:

```python
from errlore import AgentMemory
mem = AgentMemory("/app/backend/data/errlore")
mem.add_lesson("dates in extracted tables are often hallucinated",
               "Demand ISO-8601 and verify against the source document")
```

## Notes

- The Filter never fakes reinforcement: outcomes are only recorded when you
  click the feedback button. No feedback — no trust/confidence movement.
- `embeddings: true` valve enables semantic lesson retrieval
  (requires `errlore[embeddings]` in the requirements line).
