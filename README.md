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
- **Known issues** -- per-model weakness tracking ("gpt-5.6 keeps hallucinating dates in
  extraction tasks") injected as warnings.
- **Trust** *(experimental)* -- Bayesian per-model, per-domain trust weights: a starting
  point for which model to pick per job, based on observed outcomes. Needs a spread of
  real outcomes to separate models; shipped, but not yet proven on production traffic.
- **Closed loop** -- errlore tracks whether an injected lesson actually helped and
  reinforces or decays it automatically.

Embedded, file-based (JSONL), no server, no database, no API keys required.
Works fully offline. Your data never leaves your machine.

## Who it's for

errlore isn't memory for everything — it's memory for **failures**. It shines
wherever an agent repeats the *same class* of mistake:

- **Coding agents** (Claude Code, Cursor, SWE agents) that keep re-introducing
  the same bug or forgetting a project convention across sessions.
- **Extraction pipelines** (PDFs, invoices, contracts) that hallucinate the
  same date format, rounding rule, or schema field every week.
- **Any repeated-failure workflow** where a fix should stick the first time,
  not be re-discovered on every run.

It fixes what the model doesn't *know* (a convention, a gotcha), not what it
*can't do* — see the benchmark below.

## Quickstart (< 5 minutes)

```bash
pip install errlore
```

```python
from errlore import AgentMemory

mem = AgentMemory("./agent_memory")

# 1. Agent failed -- record it
err_id = mem.log_error("gpt-5.6", "extraction", error="hallucinated dates")

# 2. You fixed it -- extract a lesson
mem.resolve(err_id, "Added date format validation",
            lesson="For date extraction, demand ISO-8601 and verify against source")

# 3. Next similar task -- lessons + known issues injected automatically
inj = mem.inject_for("extract dates from contract", model="gpt-5.6",
                      task_type="extraction")
prompt = f"Your task: extract dates\n{inj.text}"
print(prompt)

# 4. Close the loop -- did the lesson help?
mem.report_outcome(inj, success=True)

# 5. Check stats
print(mem.stats())
# {'errors_total': 1, 'errors_resolved': 1, 'errors_unresolved': 0,
#  'lessons_total': 1, 'lessons_applied': 1, 'pending_injections': 0,
#  'trust': {'gpt-5.6': 0.5522...}}
```

No API keys needed. errlore itself never calls any LLM -- it manages local
JSONL files and does text matching. LLM calls are yours to make (or not).

## Does it actually reduce errors?

For the class of errors memory can fix — yes, and here's the honest version.
Paired A/B (`benchmarks/bench_error_reduction.py`): the same model
(claude-haiku-4-5) runs 96 tasks twice, with and without errlore injection.
Deterministic validators, no LLM judges; raw outputs committed in
[benchmarks/results/error_reduction/](benchmarks/results/error_reduction/) so
you can recompute every number.

| arm | failures | fail rate |
|-----|----------|-----------|
| A: plain | 63/96 | 65.6% |
| B: with errlore | 20/96 | 20.8% |

Exact McNemar over all 96 pairs: p = 1.8e-09 (49 pairs fixed, 6 broken).
Split by error class:

- **Knowledge-gap errors** (workspace conventions: date formats, ID
  normalization, rounding rules, CSV column order): 46/48 -> 0/48. The model
  can't know a convention it was never told, so arm A fails almost by
  construction; the result shows errlore **captures the fix once and re-supplies
  it** on the next similar task, end to end. That store-and-inject loop is the
  claim — not that memory teaches skills.
- **Capability-gap errors** (letter counting, string reversal): 17/48 ->
  20/48 -- errlore did **not** help and slightly hurt. Memory fixes what the
  model doesn't know, not what it can't do.

**Reproduced across 5 independent runs** (two on the default seed 5 days apart,
plus three fresh RNG seeds — different task instances). Every run: overall
reduction **66.7–69.8%**, exact McNemar **p between 8.4e-12 and 1.8e-9**,
knowledge-gap reduction **95–100%**, capability-gap **−12% to 0%** (no help).
**Cross-model:** the same grid on **gemma-4-31b** (a different model family)
lands at **70.0% reduction** (66.7% → 20.0%, p = 2.6e-13), knowledge-gap 83%,
capability-gap −20% — same effect, same honest boundary. **Task-generality:**
two fresh realistic-convention families (an arbitrary internal status enum and
a non-standard git branch convention) go **100% → 0%** on both models.
Full table + per-run reports:
[benchmarks/results/REPRODUCIBILITY_2026-07-11.md](benchmarks/results/REPRODUCIBILITY_2026-07-11.md).

**Caveats, up front:** temperature 0 still leaves LLM output slightly
non-deterministic, so exact fine-grained counts vary run to run — the large
knowledge-gap effect is robust across all five runs; the capability-gap delta
stays within noise. The knowledge-gap task families use conventions the model
demonstrably can't guess, which is the point — but it means the headline is
"the loop works," not "90% fewer errors everywhere."

Reproduce: `python benchmarks/bench_error_reduction.py --backend anthropic`
(needs an Anthropic API key; task families and validators ship in the repo).

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

**Claude Code** — one command wires up failure-memory across sessions:

```bash
errlore init claude-code            # or: --project for this repo only
```

Failed Bash commands become lessons; every new session is briefed on past
pitfalls. See [examples/claude-code/](examples/claude-code/).

| Provider    | Example                                              |
|-------------|------------------------------------------------------|
| Claude Code | [examples/claude-code/](examples/claude-code/) — hooks, `errlore init claude-code` |
| Open WebUI  | [integrations/openwebui/](integrations/openwebui/) — memory Filter + feedback Action |
| OpenAI      | [examples/openai_agent.py](examples/openai_agent.py) |
| Anthropic   | [examples/anthropic_agent.py](examples/anthropic_agent.py) |
| LangChain   | [examples/langchain_agent.py](examples/langchain_agent.py) |

The SDK examples run offline with `python examples/<name>.py` (mock responses,
no API keys). Set `use_api=True` to call real models.

### CLI

`pip install errlore` also installs an `errlore` command:

```bash
errlore init claude-code   # install Claude Code hooks + settings
errlore stats              # memory stats for a data dir (--data-dir)
errlore lessons            # list stored lessons
errlore --version
```

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

## Security

A lesson is **trusted prompt content by design** — it is injected into your
prompts and reaches the model. So:

- **Do not ingest lessons from untrusted sources without review.** Treat lesson
  capture like a code review, not like user input. A malicious lesson is a
  prompt-injection vector — and this is the real control, not the sanitizer.
- **What the sanitizer does (and does not) do.** The lesson *pattern* passes
  `sanitize_lesson_text`: it strips raw-JSON/code-fence *noise* and caps length
  so log blobs don't pollute the prompt. It is a noise filter, **not** an
  injection defense — it does not neutralize natural-language instructions, and
  the *solution* text is stored as you author it (so it can hold real code).
  Don't rely on it to make untrusted lessons safe.
- You control what becomes a lesson (`resolve(..., lesson=...)` /
  `add_lesson(...)`); nothing is auto-promoted from raw model output.

Report security issues to the address in [SECURITY.md](SECURITY.md).

## Scale & limits (honest)

errlore is built for **one process, thousands of lessons** — a single agent or
a coding-agent session, not a high-throughput fleet. Know the edges:

- **`injections.jsonl` grows unbounded.** `report_outcome` scans the whole
  ledger each call, so at very high injection volumes it slows down (roughly
  linear in total injections). Fine for interactive/agent use; log compaction
  is the next roadmap item. If you don't need the reinforcement loop, you can
  ignore `report_outcome` and the file stays small.
- **Single-process by default.** The lesson/error stores use cross-process file
  locks and are safe to share, but the **trust engine and the optional vector
  index are not cross-process safe** — two processes writing `trust.json` /
  `vectors.npy` concurrently can clobber each other (last-writer-wins). Run one
  writer, or give each process its own `data_dir`. Multi-agent shared memory is
  on the roadmap.
- **Embeddings index rebuild is O(n²) over many adds** — building a fresh index
  over a large existing lesson store is slow the first time (then incremental).
- Concurrency is tested across threads; **multi-process** stress is not yet in
  the suite.

None of these bite at the scale errlore targets today; they're stated so you
can decide, not discover.

## Roadmap

- [ ] Log compaction for injections journal
- [ ] Async API (`alog_error`, `ainject_for`, etc.)
- [ ] Multi-agent shared memory (multiple agents, one lesson store)
- [ ] Lesson clustering and auto-summarization
- [ ] Dashboard / CLI for browsing lessons and trust weights
- [ ] Export/import for lesson sharing between projects

## License

MIT
