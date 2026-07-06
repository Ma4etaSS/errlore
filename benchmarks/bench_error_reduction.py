#!/usr/bin/env python3
"""Error-reduction A/B benchmark: does errlore lesson injection reduce
repeated mistakes?

Task families were selected via a small difficulty probe so the target
model has a NON-ZERO baseline error rate (a precondition for measuring
error reduction; a model that never errs has nothing to reduce).

Protocol (paired, deterministic validators, no LLM judges):

  Pass 1 (seed): run SEED tasks with a plain prompt. Every failure is
      logged into errlore; each failing *family* gets its pre-authored
      corrective lesson via resolve(). Lessons are authored below, in this
      file, BEFORE any pass-2 output is seen — mirroring the real workflow
      where a human fixes a failure once.
  Pass 2 (test): run TEST tasks (same families, different instances) twice:
      arm A -- plain prompt (control)
      arm B -- prompt + errlore inject_for() block
      Same model, temperature 0, same order.

  Metric: per-arm failure rate on TEST tasks + exact McNemar on the paired
  outcomes. Raw model outputs are dumped to JSONL for independent audit.

Run:
  CEREBRAS_API_KEY=... python benchmarks/bench_error_reduction.py
  (options: --model, --families, --pilot)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from errlore import AgentMemory

RNG_SEED = 20260706
SEED_PER_FAMILY = 6
TEST_PER_FAMILY = 12

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

# ---------------------------------------------------------------------------
# Task families. Each generates (prompt, expected) pairs with deterministic
# string validators. Lessons are the one-time human fix for that error class.
# ---------------------------------------------------------------------------


@dataclass
class Task:
    family: str
    prompt: str
    expected: str


def fam_mult4(rng: random.Random, n: int) -> list[Task]:
    tasks = []
    for _ in range(n):
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        tasks.append(Task(
            "mult4",
            f"Compute {a}*{b}. You may show working, but the LAST LINE of "
            f"your reply must be exactly the integer result alone - no "
            f"thousands separators, no punctuation.",
            str(a * b),
        ))
    return tasks


_WORDS = ["kaleidoscope", "murmuration", "photosynthesis", "labyrinth",
          "quintessential", "juxtaposition", "serendipity", "onomatopoeia",
          "perpendicular", "hummingbird", "thunderstorm", "wheelbarrow",
          "grasshopper", "lighthouse", "watermelon", "caterpillar"]


def fam_nth_char(rng: random.Random, n: int) -> list[Task]:
    tasks = []
    for _ in range(n):
        s = rng.choice(_WORDS) + rng.choice("-._&") + rng.choice(_WORDS)
        pos = rng.randint(9, len(s) - 3)
        tasks.append(Task(
            "nth_char",
            f'What is character number {pos} (1-indexed, counting every '
            f'character including punctuation) of the string "{s}"? '
            f"You may show working, but the LAST LINE of your reply must be "
            f"exactly that single character alone.",
            s[pos - 1],
        ))
    return tasks


def fam_letter_sent(rng: random.Random, n: int) -> list[Task]:
    pool = ["the", "quick", "brown", "river", "jumps", "over", "seven",
            "lazy", "green", "turtles", "under", "bright", "summer",
            "evening", "skies", "while", "children", "gather", "berries",
            "near", "wooden", "bridges", "watching", "herons", "gliding"]
    tasks = []
    for _ in range(n):
        words = [rng.choice(pool) for _ in range(rng.randint(11, 14))]
        sent = " ".join(words)
        ch = rng.choice("erns")
        tasks.append(Task(
            "letter_sent",
            f'How many times does the letter "{ch}" appear in this sentence: '
            f'"{sent}"? You may show working, but the LAST LINE of your '
            f'reply must be exactly the integer alone.',
            str(sent.count(ch)),
        ))
    return tasks


def fam_reverse(rng: random.Random, n: int) -> list[Task]:
    tasks = []
    for _ in range(n):
        s = rng.choice(_WORDS) + rng.choice("-._") + rng.choice(_WORDS)
        tasks.append(Task(
            "reverse",
            f'Reverse the string "{s}" exactly, character by character. '
            f"You may show working, but the LAST LINE of your reply must be "
            f"exactly the reversed string alone.",
            s[::-1],
        ))
    return tasks




# --- Knowledge-gap families: the failure is a missing WORKSPACE CONVENTION,
# not a capability deficit. Without memory the model cannot know the rule;
# after one failure+fix the lesson supplies it. This mirrors the product's
# core claim (schema contracts, environment quirks, domain conventions).


def fam_log_ts(rng: random.Random, n: int) -> list[Task]:
    tasks = []
    for _ in range(n):
        y, mo, d = rng.randint(2023, 2026), rng.randint(1, 12), rng.randint(1, 28)
        h, mi = rng.randint(0, 23), rng.randint(0, 59)
        iso = f"{y}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:00Z"
        tasks.append(Task(
            "log_ts",
            f"Convert this timestamp to our internal log format: {iso}. "
            f"The LAST LINE of your reply must be the formatted timestamp alone.",
            f"{d:02d}|{mo:02d}|{y}@{h:02d}:{mi:02d}",
        ))
    return tasks


def fam_id_norm(rng: random.Random, n: int) -> list[Task]:
    first = ["Anna", "Boris", "Clara", "Dmitri", "Elena", "Felix", "Greta", "Hugo"]
    last = ["Koval", "Smith", "Weber", "Rossi", "Novak", "Braun", "Lang", "Mora"]
    tasks = []
    for _ in range(n):
        raw = f"{rng.choice(first)}-{rng.choice(last)}-{rng.randint(10, 99)}"
        tasks.append(Task(
            "id_norm",
            f'Normalize this raw user id for our system: "{raw}". '
            f"The LAST LINE of your reply must be the normalized id alone.",
            "u_" + raw.lower().replace("-", ""),
        ))
    return tasks


def fam_round_rule(rng: random.Random, n: int) -> list[Task]:
    tasks = []
    for _ in range(n):
        cents = rng.randint(1000, 99999)
        third = rng.choice([3, 6, 7, 9])
        price = cents / 100 + third / 1000  # e.g. 123.456
        expected = f"{int(price * 100) / 100:.2f}"  # truncate = round DOWN
        tasks.append(Task(
            "round_rule",
            f"Apply our finance rounding rule to {price:.3f} and give the "
            f"amount with 2 decimals. The LAST LINE must be the number alone.",
            expected,
        ))
    return tasks


def fam_csv_order(rng: random.Random, n: int) -> list[Task]:
    names = ["Ivy", "Max", "Zoe", "Kai", "Lea", "Tom", "Ada", "Rex"]
    tasks = []
    for _ in range(n):
        name = rng.choice(names)
        uid = rng.randint(100, 999)
        email = f"{name.lower()}{uid}@example.com"
        fields = {"name": name, "id": str(uid), "email": email}
        keys = list(fields)
        rng.shuffle(keys)
        jumbled = json.dumps({k: fields[k] for k in keys})
        tasks.append(Task(
            "csv_order",
            f"Convert this record to a CSV row in our canonical column "
            f"order: {jumbled}. The LAST LINE must be the CSV row alone.",
            f"{email},{uid},{name}",
        ))
    return tasks

# capability-gap families (model skill limits) + knowledge-gap families
# (workspace conventions) — reported separately.
FAMILIES = {
    "mult4": fam_mult4,
    "nth_char": fam_nth_char,
    "letter_sent": fam_letter_sent,
    "reverse": fam_reverse,
    "log_ts": fam_log_ts,
    "id_norm": fam_id_norm,
    "round_rule": fam_round_rule,
    "csv_order": fam_csv_order,
}

KNOWLEDGE_GAP = {"log_ts", "id_norm", "round_rule", "csv_order"}

# Pre-authored corrective lessons (the "human fix", written before pass 2).
LESSONS = {
    "mult4": "Never answer multi-digit multiplication from intuition. Break it into partial products and add them carefully. Verify: the last digit of the result must equal (last digit of a * last digit of b) mod 10. Then put the final integer ALONE on the last line - no commas, no words after it.",
    "nth_char": "Write out the string with numbered positions (1,2,3,...), counting EVERY character including hyphens and punctuation, then pick exactly the requested position. Put that single character ALONE on the last line.",
    "letter_sent": "Go word by word and count the target letter in each word, keeping a running total across ALL words. Then put the final integer ALONE on the last line.",
    "reverse": "Build the reversal by writing characters from the END one at a time. Verify lengths match and the first output char equals the last input char. Then put the reversed string ALONE on the last line.",
    "log_ts": "Our internal log timestamp format is DD|MM|YYYY@HH:mm (day first, pipe separators, @ before the 24h time, no seconds, no timezone). Example: 2024-03-05T14:07:00Z -> 05|03|2024@14:07.",
    "id_norm": "Our user-id normalization rule: lowercase everything, remove all dashes, then add the prefix u_. Example: Anna-Koval-42 -> u_annakoval42.",
    "round_rule": "Our finance rule ALWAYS truncates (rounds toward zero) to 2 decimals - never round half up or to even. Example: 123.456 -> 123.45, and 99.999 -> 99.99.",
    "csv_order": "Our canonical CSV column order is: email,id,name (regardless of the order fields appear in the source record). No spaces after commas, no quotes.",
}

SYSTEM = "You are a precise assistant. Follow the required output format exactly."


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate(task: Task, output: str) -> bool:
    out = output.strip()
    if task.expected == "__JSON5__":
        try:
            if out.startswith("```"):
                out = out.strip("`")
                out = out[out.find("{"):out.rfind("}") + 1]
            obj = json.loads(out)
        except (json.JSONDecodeError, ValueError):
            return False
        fixes = obj.get("fixes")
        return (
            isinstance(obj, dict) and list(obj.keys()) == ["fixes"]
            and isinstance(fixes, list) and len(fixes) == 5
            and all(isinstance(x, str) and x.strip() for x in fixes)
            and len(set(fixes)) == 5
        )
    # exact-match families: the contract is "last line = answer alone"
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    last = (lines[-1] if lines else "").rstrip(".").strip().strip('"')
    return last == task.expected


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------


def make_client(backend: str) -> tuple[OpenAI, str]:
    if backend == "cerebras":
        key = os.environ.get("CEREBRAS_API_KEY")
        if not key:
            sys.exit("CEREBRAS_API_KEY is required")
        return OpenAI(base_url="https://api.cerebras.ai/v1", api_key=key), "gemma-4-31b"
    if backend == "anthropic":
        # Uses the Anthropic SDK directly (no OpenAI-compatible endpoint).
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("ANTHROPIC_API_KEY is required")
        return None, "claude-haiku-4-5"  # client built lazily in ask()
    if backend == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY is required")
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=key,
        ), "gemini-2.5-flash-lite"
    sys.exit(f"unknown backend {backend}")


_anthropic_client = None


def _ask_anthropic(model: str, system: str, prompt: str) -> str:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    for attempt in range(6):
        try:
            resp = _anthropic_client.messages.create(
                model=model,
                max_tokens=500,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return next((b.text for b in resp.content if b.type == "text"), "")
        except anthropic.RateLimitError:
            if attempt == 5:
                raise
            time.sleep(31)
    raise RuntimeError("unreachable")


def ask(client: OpenAI, model: str, prompt: str, extra_system: str = "") -> str:
    from openai import RateLimitError

    system = SYSTEM + (("\n\n" + extra_system) if extra_system else "")
    if client is None:  # anthropic backend
        return _ask_anthropic(model, system, prompt)
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=400,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            if attempt == 5:
                raise
            time.sleep(31)  # free-tier RPM window
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (binomial)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini")
    ap.add_argument("--pilot", action="store_true", help="2 families, 3 test each")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    client, model = make_client(args.backend)
    rng = random.Random(RNG_SEED)

    fams = dict(FAMILIES)
    seed_n, test_n = SEED_PER_FAMILY, TEST_PER_FAMILY
    if args.pilot:
        fams = {k: fams[k] for k in ("mult4", "nth_char")}
        seed_n, test_n = 3, 3

    seed_tasks: list[Task] = []
    test_tasks: list[Task] = []
    for _name, gen in fams.items():
        all_t = gen(rng, seed_n + test_n)
        seed_tasks += all_t[:seed_n]
        test_tasks += all_t[seed_n:]

    workdir = Path(tempfile.mkdtemp(prefix="errlore_ab_"))
    mem = AgentMemory(workdir / "memory")
    raw_log = open(workdir / "raw_outputs.jsonl", "w")

    def record(phase: str, arm: str, task: Task, output: str, ok: bool) -> None:
        raw_log.write(json.dumps({
            "phase": phase, "arm": arm, "family": task.family,
            "prompt": task.prompt, "expected": task.expected,
            "output": output, "ok": ok,
        }, ensure_ascii=False) + "\n")

    # ---- Pass 1: seed (plain prompts; failures become lessons) ----
    fam_fail: dict[str, int] = {}
    print(f"[pass1] {len(seed_tasks)} seed tasks, model={model}")
    for t in seed_tasks:
        out = ask(client, model, t.prompt)
        ok = validate(t, out)
        record("seed", "plain", t, out, ok)
        if not ok:
            fam_fail[t.family] = fam_fail.get(t.family, 0) + 1
            err_id = mem.log_error(model, t.family, f"WrongAnswer: {out[:120]}")
            mem.resolve(err_id, "authored corrective lesson", lesson=LESSONS[t.family])
    print(f"[pass1] failures by family: {fam_fail or 'none'}")
    active = set(fam_fail)
    if not active:
        print("[pass1] model aced the seed set — no lessons to test; "
              "harden the task families before drawing conclusions.")
        return 1

    # ---- Pass 2: paired test ----
    results = []  # (family, ok_A, ok_B)
    print(f"[pass2] {len(test_tasks)} test tasks x 2 arms")
    for t in test_tasks:
        out_a = ask(client, model, t.prompt)
        ok_a = validate(t, out_a)
        record("test", "A_plain", t, out_a, ok_a)

        inj = mem.inject_for(t.prompt, model=model, task_type=t.family)
        out_b = ask(client, model, t.prompt, extra_system=inj.text)
        ok_b = validate(t, out_b)
        record("test", "B_errlore", t, out_b, ok_b)
        mem.report_outcome(inj, ok_b)
        results.append((t.family, ok_a, ok_b))

    raw_log.close()

    # ---- Report ----
    n = len(results)
    fail_a = sum(1 for _, a, _b in results if not a)
    fail_b = sum(1 for _, _a, b in results if not b)
    b_disc = sum(1 for _, a, b in results if a and not b)   # B fails where A passed
    c_disc = sum(1 for _, a, b in results if not a and b)   # B fixes A's failure
    p = mcnemar_exact_p(b_disc, c_disc)

    lines = [
        f"# errlore error-reduction A/B — model {model}",
        "",
        f"tasks (test): {n} | families active (had seed failures): {sorted(active)}",
        "",
        "| arm | failures | fail rate |",
        "|---|---|---|",
        f"| A plain | {fail_a}/{n} | {fail_a/n:.1%} |",
        f"| B errlore | {fail_b}/{n} | {fail_b/n:.1%} |",
        "",
        f"discordant pairs: errlore fixed {c_disc}, errlore broke {b_disc}",
        f"exact McNemar p = {p:.4g}",
    ]
    if fail_a:
        lines.append(f"repeat-error reduction: {(fail_a - fail_b) / fail_a:.1%}")
    for label, group in (("KNOWLEDGE-GAP (workspace conventions)",
                          [r for r in results if r[0] in KNOWLEDGE_GAP]),
                         ("CAPABILITY-GAP (model skill limits)",
                          [r for r in results if r[0] not in KNOWLEDGE_GAP])):
        if not group:
            continue
        gn = len(group)
        ga = sum(1 for _, a, _b in group if not a)
        gb = sum(1 for _, _a, b in group if not b)
        red = f" | reduction {(ga - gb) / ga:.0%}" if ga else ""
        lines += ["", f"### {label}: A {ga}/{gn} -> B {gb}/{gn}{red}"]

    lines += ["", "per-family (fail A -> fail B):"]
    for fam in sorted(fams):
        fa = sum(1 for f, a, _ in results if f == fam and not a)
        fb = sum(1 for f, _, b in results if f == fam and not b)
        marker = " *lesson active*" if fam in active else ""
        lines.append(f"- {fam}: {fa} -> {fb}{marker}")
    lines += ["", f"raw outputs: {workdir}/raw_outputs.jsonl",
              f"errlore stats: {mem.stats()}"]
    report = "\n".join(lines)
    print("\n" + report)
    out_path = Path(args.out) if args.out else workdir / "report.md"
    out_path.write_text(report)
    print(f"\n[saved: {out_path}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
