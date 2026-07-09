#!/usr/bin/env python3
"""OpenAI integration example -- errlore learning loop with GPT-4o.

Demonstrates the full errlore lifecycle:
  1. log_error   -- record a failure
  2. resolve     -- extract a lesson from the fix
  3. inject_for  -- enrich the next prompt with past lessons + known issues
  4. report_outcome -- close the loop (reinforce/decay lessons, update trust)

Run offline (no API key needed):
    python examples/openai_agent.py

The ``if __name__`` block uses a mock response so the errlore part
runs end-to-end without network access.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from errlore import AgentMemory

# Model is just a label to errlore (it never calls the API itself). Override
# with your own; the default is a small, cheap, widely-available option.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# -- Agent wrapper -----------------------------------------------------------

def ask_openai(prompt: str, system: str) -> str:
    """Call the OpenAI Chat Completions API.

    Requires OPENAI_API_KEY in the environment.
    """
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    text = resp.choices[0].message.content
    return text or ""


def run_with_errlore(
    mem: AgentMemory,
    task: str,
    *,
    use_api: bool = False,
    mock_response: str = "",
) -> str:
    """Run a task through errlore-augmented OpenAI agent.

    When *use_api* is False (default), returns *mock_response* instead of
    calling OpenAI -- useful for offline demos and tests.
    """
    # 1. Build injection: past lessons + known issues for this model
    inj = mem.inject_for(task, model=MODEL)

    system = "You are a helpful assistant."
    if inj.text:
        system += "\n\n" + inj.text

    # 2. Call the model (or mock)
    if use_api:
        try:
            answer = ask_openai(task, system)
        except Exception as exc:
            # Record the failure so errlore learns
            err_id = mem.log_error(MODEL, "general", error=exc)
            mem.resolve(err_id, f"API error: {exc}")
            mem.report_outcome(inj, success=False)
            raise
    else:
        answer = mock_response

    # 3. Evaluate and close the loop
    success = len(answer) > 0 and "error" not in answer.lower()
    mem.report_outcome(inj, success=success)

    return answer


# -- Offline demo ------------------------------------------------------------

def _demo() -> None:
    """Self-contained offline demo -- no API key needed."""
    with tempfile.TemporaryDirectory() as tmp:
        mem = AgentMemory(Path(tmp) / "agent_memory")

        # Simulate a past failure and its resolution
        err_id = mem.log_error(
            MODEL, "extraction",
            error="Hallucinated dates: returned 2025-13-45",
        )
        mem.resolve(
            err_id,
            "Added ISO-8601 format validation",
            lesson="For date extraction, demand ISO-8601 format and verify "
                   "each date against the source document before returning.",
        )

        # Now run a similar task -- errlore injects the lesson automatically
        task = "Extract all dates from the attached contract."
        answer = run_with_errlore(
            mem, task,
            mock_response="2025-06-15 (effective date), 2026-06-14 (expiry)",
        )

        # Print what happened
        inj = mem.inject_for(task, model=MODEL)
        print("=== errlore injection for next task ===")
        print(inj.text or "(empty -- no lessons yet)")
        print()
        print(f"=== Agent answer (mock) ===\n{answer}")
        print()

        # Show stats
        stats = mem.stats()
        print("=== Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    _demo()
