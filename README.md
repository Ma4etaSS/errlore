# errlore

**Memory for AI agents that learns from failures.**

*Stop the second mistake, not just the first.*

![errlore demo: monday failure becomes a lesson, tuesday's prompt gets it injected](https://errlore.com/demo.gif)

[![CI](https://github.com/Ma4etaSS/errlore/actions/workflows/ci.yml/badge.svg)](https://github.com/Ma4etaSS/errlore/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Extracted from a 324K LOC production multi-LLM orchestration system, keeping the one part that demonstrably worked: the error-memory loop that made agents stop repeating mistakes.

Your agent keeps making the same mistakes. errlore fixes that:

- **Lessons** -- every resolved failure becomes a lesson; relevant lessons are injected
  into the prompt for similar future tasks.
- **Known issues** -- per-model weakness tracking ("gpt-5.5 keeps hallucinating dates in
  extraction tasks") injected as warnings.
- **Trust** *(experimental)* -- Bayesian per-model, per-domain trust weights: a starting
  point for which model to pick per job, based on observed outcomes. Needs a spread of
  real outcomes to separate models; shipped, but not yet proven on production traffic.
- **Closed loop** -- errlore tracks whether an injected lesson actually helped and
  reinforces or decays it automatically.

Embedded, file-based (JSONL), no server, no database, no API keys required.
Works fully offline. Your data never leaves your machine.

## Quickstart (< 5 minutes)

```bash
pip install errlore
```

```python
from errlore import AgentMemory

mem = AgentMemory("./agent_memory")

# 1. Agent failed -- record it
err_id = mem.log_error("gpt-5.5", "extraction", error="hallucinated dates")

# 2. You fixed it -- extract a lesson
mem.resolve(err_id, "Added date format validation",
            lesson="For date extraction, demand ISO-8601 and verify against source")

# 3. Next similar task -- lessons + known issues injected automatically
inj = mem.inject_for("extract dates from contract", model="gpt-5.5",
                      task_type="extraction")
prompt = f"Your task: extract dates\n{inj.text}"
print(prompt)

# 4. Close the loop -- did the lesson help?
mem.report_outcome(inj, success=True)

# 5. Check stats
print(mem.stats())
# {'errors_total': 1, 'errors_resolved': 1, 'errors_unresolved': 0,
#  'lessons_total': 1, 'lessons_applied': 1, 'pending_injections': 0,
#  'trust': {'gpt-5.5': 0.5522...}}
```

No API keys needed. errlore itself never calls any LLM -- it manages local
JSONL files and does text matching. LLM calls are yours to make (or not).

## Does it actually reduce errors?

Yes -- for the class of errors memory can fix. Paired A/B benchmark
(`benchmarks/bench_error_reduction.py`): the same model (claude-haiku-4-5)
runs 96 tasks twice, with and without errlore injection. Deterministic
validators, no LLM judges; raw outputs committed in
[benchmarks/results/error_reduction/](benchmarks/results/error_reduction/).

| arm | failures | fail rate |
|-----|----------|-----------|
| A: plain | 63/96 | 65.6% |
| B: with errlore | 20/96 | 20.8% |

Exact McNemar over all 96 pairs: p = 1.8e-09 (49 pairs fixed, 6 broken).
Split by error class:

- **Knowledge-gap errors** (workspace conventions: date formats, ID
  normalization, rounding rules, CSV column order): **46/48 -> 0/48, a 100%
  reduction.** The model didn't know the convention; a lesson told it.
- **Capability-gap errors** (letter counting, string reversal): 17/48 ->
  20/48 -- errlore did **not** help and slightly hurt. Memory fixes what the
  model doesn't know, not what it can't do.

Reproduce: `python benchmarks/bench_error_reduction.py` (needs an Anthropic
API key; the task families and validators ship in the repo).

## How it works

errlore runs three reinforcement loops around your agent:

### 1. Lesson loop

```
Agent fails  -->  log_error()  -->  resolve() + lesson
                                        |
Agent runs   <--  inject_for() <--------+
     |
     +--> report_outcome(success=True)  -->  lesson confidence +0.1
     +--> report_outcome(success=False) -->  lesson confidence -0.1
```

Lessons with high confidence surface first. Unused lessons decay over time.

### 2. Known-issue loop

Per-model, per-task-type error tracking. When a model has failed on a task
type before, `inject_for` adds a warning block to the prompt. Separate from
lessons: lessons are *solutions*, known issues are *warnings*.

### 3. Trust loop *(experimental)*

Bayesian per-model weights with adaptive learning rate, cold-start blending,
entropy enforcement, and temporal decay.  After enough observations, call
`mem.best_model("code_generation")` to pick the model that historically
performs best on that domain.

> **Status: experimental.** The engine is tested and works, but discrimination
> between models only emerges from a *spread* of real outcomes over time — feed
> it a stream that is mostly successes and every model converges near the cap.
> Treat `best_model()` as a hint to validate, not a proven router yet. The
> lesson + known-issue loops above are the proven core (see the A/B benchmark).

## Semantic retrieval (optional)

By default, errlore finds relevant lessons via word overlap (zero
dependencies). For higher recall on paraphrased queries, enable embedding
search:

```bash
pip install errlore[embeddings]   # installs fastembed + numpy
```

```python
mem = AgentMemory("./agent_memory", embeddings=True)
```

> The embedding model (~120 MB ONNX) is downloaded once on first use, then
> runs locally with no further network calls. The core (word-overlap) stays
> fully offline and dependency-free.

### Benchmark (adversarial paraphrasing)

Tested on 40 lessons with adversarially paraphrased queries
(`benchmarks/bench_retrieval.py`):

| Metric    | word-overlap | embeddings |
|-----------|-------------|------------|
| recall@1  | 0.000       | 0.375      |
| recall@3  | 0.000       | 0.575      |
| recall@5  | 0.000       | 0.675      |
| MRR       | 0.000       | 0.488      |

The gold set is intentionally adversarial (queries share few literal words
with the lesson text), which is why word-overlap scores zero.  On natural
queries with shared vocabulary, word-overlap works fine.

## Integrations

errlore is framework-agnostic. It produces a text block; you put it in the
system prompt.

| Provider   | Example                                              |
|------------|------------------------------------------------------|
| OpenAI     | [examples/openai_agent.py](examples/openai_agent.py) |
| Anthropic  | [examples/anthropic_agent.py](examples/anthropic_agent.py) |
| LangChain  | [examples/langchain_agent.py](examples/langchain_agent.py) |

All examples run offline with `python examples/<name>.py` (mock responses,
no API keys). Set `use_api=True` to call real models.

## API overview

The main entry point is `AgentMemory`. All other classes are internal --
you only need them for advanced use.

| Method / Property             | Description                                    |
|-------------------------------|------------------------------------------------|
| `log_error(model, task_type, error)` | Record an error. Returns error ID.      |
| `resolve(err_id, resolution, lesson)` | Mark error fixed, extract a lesson.    |
| `inject_for(task, model)`     | Build prompt injection (lessons + warnings).   |
| `report_outcome(inj, success)` | Close the loop: reinforce lessons, update trust.|
| `add_lesson(pattern, solution)` | Add a lesson directly (sanitized).            |
| `lessons(limit)`              | List all lessons (sorted by confidence).       |
| `best_model(domain)`          | Model with the highest trust weight *(experimental)*. |
| `model_penalty(model, task_type)` | Error-history penalty `[0, 1]`.            |
| `pending_injections()`        | Injections not yet reported.                   |
| `stats()`                     | Aggregate counts + trust weights.              |
| `.trust`                      | Access the underlying `TrustEngine` (or None). |

### Supporting classes (advanced)

| Class             | Purpose                                    |
|-------------------|--------------------------------------------|
| `LessonStore`     | Low-level lesson CRUD + search.            |
| `TrustEngine`     | Bayesian trust weights with persistence.   |
| `FeedbackSignal`  | Typed quality signal for trust updates.    |
| `Injection`       | Dataclass returned by `inject_for`.        |

## Data & privacy

- All data is stored in local JSONL files in the directory you specify.
- Nothing is sent to any server.  errlore itself makes zero network calls.
- Works fully offline -- no API keys, no accounts, no telemetry.
- Files: `errors.jsonl`, `lessons.jsonl`, `injections.jsonl`, `trust.json`,
  `model_accuracy.jsonl`.
- Sidecar files (auto-managed): `*.idx` (byte-offset index), `*.lock`
  (filelock), `vectors.npy` (embedding vectors), `vector_meta.json`
  (embedding metadata), `trust.json` (trust engine state).

## Roadmap

- [ ] Log compaction for injections journal
- [ ] Async API (`alog_error`, `ainject_for`, etc.)
- [ ] Multi-agent shared memory (multiple agents, one lesson store)
- [ ] Lesson clustering and auto-summarization
- [ ] Dashboard / CLI for browsing lessons and trust weights
- [ ] Export/import for lesson sharing between projects

## License

MIT
