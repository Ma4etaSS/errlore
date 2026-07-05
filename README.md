# errlore

**Memory for AI agents that learns from failures.**

Your agent keeps making the same mistakes. errlore fixes that:

- **Lessons** — every resolved failure becomes a lesson; relevant lessons are injected
  into the prompt for similar future tasks.
- **Known issues** — per-model weakness tracking ("gpt-x keeps hallucinating dates in
  extraction tasks") injected as warnings.
- **Trust** — Bayesian per-model, per-domain trust weights: know which model to pick
  for which job, based on observed outcomes.
- **Closed loop** — errlore tracks whether an injected lesson actually helped and
  reinforces or decays it automatically.

Embedded, file-based (JSONL), no server, no database, no API keys required.
Your data never leaves your machine.

```python
from errlore import AgentMemory

mem = AgentMemory("./agent_memory")

# agent failed
err_id = mem.log_error("gpt-4o", "extraction", error="hallucinated dates")
mem.resolve(err_id, "Added date format validation",
            lesson="For date extraction, demand ISO-8601 and verify against source")

# next similar task — lessons + KNOWN ISSUES injected automatically
inj = mem.inject_for("extract dates from contract", model="gpt-4o")
prompt = f"Your task: extract dates from contract\n{inj.text}"

# close the loop — reinforces lessons that helped, updates trust weights
mem.report_outcome(inj, success=True)
```

## Status

Alpha — extracted from a production multi-LLM orchestration system (324K LOC),
keeping the one part that demonstrably worked.

## License

MIT
