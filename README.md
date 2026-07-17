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
- **Harm gate** -- injecting lessons also *breaks* some previously-passing tasks (we
  measured 12–15%). errlore tracks each lesson's failures separately and withholds one
  from injection once a Beta-Binomial bar says its harm rate is credibly too high — so a
  bad lesson can't keep hurting you. A static conventions file can't do this.
- **Warning tier** *(validator-less surfaces)* -- no oracle? Run the prompt twice and
  pass both to `check_consistency`: disagreement flags "likely wrong" at ~86% precision.
  Honestly one-sided — a *stable* result is explicitly **not** verification. Cheap
  wrong-answer detector, never a correctness guarantee.
- **Shadow mode** *(validator-equipped surfaces)* -- verify a lesson before it graduates.
  A counterfactual run (never the user's output) re-tests baseline vs injected against
  your validator; two Beta posteriors decide `promote` / `hold` / `quarantine`. Lessons
  that clear the bar graduate into your prompt/docs with an evidence trail. See
  [docs/SHADOW_MODE_SPEC.md](docs/SHADOW_MODE_SPEC.md).

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

### Why only failures? (FAQ)

Because that's the slice of memory we could *prove* pays for itself. errlore
was extracted from a 324K-LOC agent system with general remember-everything
memory; when we benchmarked which memory changed outcomes, one loop survived —
failure → lesson → injection into the next similar task (67–70% repeat-error
reduction, McNemar p ≤ 2e-9). Failures are special for two reasons:

1. **They carry a built-in relevance signal** — a resolved error says exactly
   *when* the memory matters again (same task class) and *what to say* (the
   fix). General memories don't; you inject "context" and hope.
2. **Injection has a measurable cost** — lessons *break 12–15% of
   previously-passing tasks* (interference). errlore controls that per lesson
   (harm gate, shadow mode); a remember-everything memory multiplies the
   interference surface with no per-item outcome tracking to contain it.

Other memory types aren't useless — they just don't (yet) have this
verifiable structure. Longer discussion:
[issue #1](https://github.com/Ma4etaSS/errlore/issues/1).

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

**LangChain** — first-class middleware + callbacks (`pip install errlore[langchain]`):

```python
from errlore import AgentMemory
from errlore.integrations.langchain import ErrloreCallbackHandler, ErrloreMiddleware
from langchain.agents import create_agent

mem = AgentMemory("./errlore-data")
mw = ErrloreMiddleware(mem, model="gpt-5.5", task_type="agent")

agent = create_agent(model="gpt-5.5", tools=[...], middleware=[mw])
result = agent.invoke(
    {"messages": [("user", "extract the invoice dates")]},
    config={"callbacks": [ErrloreCallbackHandler(mem, model="gpt-5.5")]},
)
mw.report(success=True)   # after YOUR validation of the result
```

`ErrloreMiddleware` injects relevant lessons into the system prompt on each
run; `ErrloreCallbackHandler` auto-captures tool/LLM errors into memory —
the same capture loop as the Claude Code hooks.

| Provider    | Example                                              |
|-------------|------------------------------------------------------|
| Claude Code | [examples/claude-code/](examples/claude-code/) — hooks, `errlore init claude-code` |
| LangChain   | [src/errlore/integrations/langchain.py](src/errlore/integrations/langchain.py) — middleware + callbacks, [examples/langchain_agent.py](examples/langchain_agent.py) |
| Open WebUI  | [integrations/openwebui/](integrations/openwebui/) — memory Filter + feedback Action |
| OpenAI      | [examples/openai_agent.py](examples/openai_agent.py) |
| Anthropic   | [examples/anthropic_agent.py](examples/anthropic_agent.py) |

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
| `quarantined_lessons()`       | Lessons the harm gate withholds from injection.|
| `check_consistency(outputs)`  | Warning tier: flag likely-wrong via re-run disagreement. |
| `enqueue_counterfactual(inj, baseline)` | Shadow mode: queue a lesson's counterfactual trial. |
| `report_counterfactual_outcome(cf_id, base_ok, inj_ok)` | Close a shadow trial; update graduation. |
| `graduation_status(lesson_id)` | `promote` / `hold` / `quarantine` from shadow evidence. |
| `graduated_lessons()`         | Lessons verified ready to bake into a permanent surface. |
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
- Files: `errors.jsonl`, `lessons.jsonl`, `injections.jsonl`,
  `counterfactuals.jsonl` (shadow mode), `trust.json`, `model_accuracy.jsonl`.
- Sidecar files (auto-managed): `*.idx` (byte-offset index), `*.lock`
  (filelock), `vectors.npy` (embedding vectors), `vector_meta.json`
  (embedding metadata), `trust.json` (trust engine state).

## Security

A lesson is **trusted prompt content by design** — it is injected into your
prompts and reaches the model. So:

- **Do not ingest lessons from untrusted sources without review.** Treat lesson
  capture like a code review, not like user input. A malicious lesson is a
  prompt-injection vector — and this is the real control, not the sanitizer.
- **What the sanitizer does (and does not) do.** Both the lesson *pattern* and
  *solution* pass `sanitize_lesson_text` at the injection boundary: it strips
  raw-JSON/code-fence *noise* and control characters (ANSI/NUL), caps length,
  NFKC-normalizes (homoglyph/full-width/zero-width tricks fold to ASCII), and
  **redacts the "ignore all previous instructions" override family** — tuned to
  catch the obvious payloads without touching legitimate lessons that merely
  mention instructions or prompts. The injected block is also explicitly framed
  as reference data, not instructions. All of that is defense-in-depth, **not**
  a complete injection defense — a determined author can phrase an override no
  pattern list catches. Don't rely on it to make untrusted lessons safe; review
  is the real control.
- You control what becomes a lesson (`resolve(..., lesson=...)` /
  `add_lesson(...)`); nothing is auto-promoted from raw model output.

### Privacy mode

Error descriptions come from tool output (stderr, command lines), which can
carry credentials, emails, and addresses. With privacy mode on, every text
field is scrubbed **before it reaches disk** — so secrets never land in
`errors.jsonl`/`lessons.jsonl`, and therefore can never be re-injected into a
later prompt:

```python
mem = AgentMemory(
    "./data",
    privacy_mode=True,
    redact_patterns=[r"CUST-\d{6}"],   # optional extra regexes -> [REDACTED]
)
```

Defaults redact emails, IPv4 addresses, `Bearer` tokens, `password=...` /
`api_key: ...` pairs, and well-known key prefixes (`sk-…`, `ghp_…`,
`github_pat_…`, `AKIA…`, `xox…`) — precision over recall, so hashes and IDs a
lesson needs stay readable. For the Claude Code hooks, set
`ERRLORE_PRIVACY_MODE=1` in the environment.

Report security issues to the address in [SECURITY.md](SECURITY.md).

## Scale & limits (honest)

errlore is built for **one process, thousands of lessons** — a single agent or
a coding-agent session, not a high-throughput fleet. Know the edges:

- **`injections.jsonl` self-compacts** (since 0.3.2): once the ledger passes a
  size threshold, the heavy `issued` record of every closed handle is dropped,
  keeping only its tiny `reported` marker and any pending injections — so
  `report_outcome`'s full scan stays bounded. If you don't need the
  reinforcement loop, you can ignore `report_outcome` and the file stays small.
- **Single-process by default.** The lesson/error stores use cross-process file
  locks and are safe to share, but the **trust engine and the optional vector
  index are not cross-process safe** — two processes writing `trust.json` /
  `vectors.npy` concurrently can clobber each other (last-writer-wins). Run one
  writer, or give each process its own `data_dir`. Multi-agent shared memory is
  on the roadmap.
- **Embeddings index rebuild is O(n²) over many adds** — building a fresh index
  over a large existing lesson store is slow the first time (then incremental).
- Concurrency is tested across threads **and across real OS processes**
  (lost-update and at-most-once-report suites) for the lesson/error/injection
  stores; the trust engine and vector index remain single-writer as above.

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
