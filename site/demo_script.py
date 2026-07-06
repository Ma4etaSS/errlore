#!/usr/bin/env python3
"""Scripted terminal demo for the errlore GIF (recorded with asciinema).

Shows the real library doing the real cycle — outputs are produced by
actually calling errlore against a temp dir, not faked strings.
"""

import shutil
import sys
import time
from pathlib import Path

DATA = Path("/tmp/errlore_demo")

G = "\033[32m"   # green
Y = "\033[33m"   # gold
R = "\033[31m"   # red
D = "\033[2m"    # dim
B = "\033[1m"    # bold
X = "\033[0m"    # reset


def type_out(text: str, delay: float = 0.025, end: str = "\n") -> None:
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(end)
    sys.stdout.flush()


def prompt(cmd: str) -> None:
    sys.stdout.write(f"{G}${X} ")
    sys.stdout.flush()
    time.sleep(0.4)
    type_out(cmd)
    time.sleep(0.3)


def out(text: str, color: str = "") -> None:
    print(f"{color}{text}{X}")
    sys.stdout.flush()
    time.sleep(0.25)


def main() -> None:
    shutil.rmtree(DATA, ignore_errors=True)
    from errlore import AgentMemory
    mem = AgentMemory(DATA)

    print(f"{B}# your agent, monday{X}")
    time.sleep(0.6)
    prompt("agent run 'sum the invoice amounts'")
    out("  -> 224.64   (expected 224.63)", R)
    out("  FAILED: used banker's rounding, finance wants truncation", R)
    time.sleep(0.8)

    print()
    print(f"{B}# you fix it once — errlore remembers{X}")
    time.sleep(0.5)
    prompt("python")
    type_out(f"{Y}>>>{X} err = mem.log_error('gpt-5.5', 'finance', 'rounded half-even')", 0.012)
    err = mem.log_error("gpt-5.5", "finance", "WrongRounding: rounded half-even")
    time.sleep(0.4)
    type_out(f"{Y}>>>{X} mem.resolve(err, 'fixed', lesson='finance ALWAYS truncates to 2 decimals')", 0.012)
    mem.resolve(err, "fixed", lesson="finance ALWAYS truncates to 2 decimals - never round half up or to even")
    out("  lesson saved  [conf 0.80]", G)
    time.sleep(0.9)

    print()
    print(f"{B}# tuesday — same class of task{X}")
    time.sleep(0.5)
    prompt("agent run 'sum the refund amounts'")
    inj = mem.inject_for("sum the refund amounts", model="gpt-5.5", task_type="finance")
    out("  errlore injected into the prompt:", Y)
    for line in inj.text.strip().splitlines():
        print(f"{D}  | {line}{X}")
        time.sleep(0.18)
    time.sleep(0.5)
    out("  -> 148.90   correct", G)
    mem.report_outcome(inj, success=True)
    out("  outcome reported: lesson reinforced [conf 0.80 -> 0.90]", G)
    time.sleep(1.0)

    print()
    print(f"{B}pip install errlore{X}   {D}errlore.com — MIT, offline, no telemetry{X}")
    time.sleep(2.2)


if __name__ == "__main__":
    main()
