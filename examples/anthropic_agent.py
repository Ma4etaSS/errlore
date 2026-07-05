#!/usr/bin/env python3
"""Anthropic integration example -- errlore learning loop with Claude.

Demonstrates the full errlore lifecycle:
  1. log_error   -- record a failure
  2. resolve     -- extract a lesson from the fix
  3. inject_for  -- enrich the next prompt with past lessons + known issues
  4. report_outcome -- close the loop (reinforce/decay lessons, update trust)

Run offline (no API key needed):
    python examples/anthropic_agent.py

The ``if __name__`` block uses a mock response so the errlore part
runs end-to-end without network access.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from errlore import AgentMemory

# Current Claude lineup (2026): claude-fable-5 (most capable),
# claude-opus-4-8 (recommended default), claude-sonnet-4-6 (speed/cost balance),
# claude-haiku-4-5 (fastest). Use exact IDs as-is -- no date suffixes.
MODEL = "claude-opus-4-8"


# -- Agent wrapper -----------------------------------------------------------

def ask_anthropic(task: str, system: str) -> str:
    """Call the Anthropic Messages API.

    Requires ANTHROPIC_API_KEY in the environment.

    Note: Claude Opus 4.8 rejects sampling parameters (temperature, top_p),
    so we do not pass them.
    """
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": task}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text


def run_with_errlore(
    mem: AgentMemory,
    task: str,
    *,
    task_type: str = "general",
    use_api: bool = False,
    mock_response: str = "",
) -> str:
    """Run a task through errlore-augmented Anthropic agent.

    When *use_api* is False (default), returns *mock_response* instead of
    calling Anthropic -- useful for offline demos and tests.
    """
    # 1. Build injection: past lessons + known issues for this model
    inj = mem.inject_for(task, model=MODEL, task_type=task_type)

    system = "You are a careful analyst. Be precise and cite sources."
    if inj.text:
        system += "\n\n" + inj.text

    # 2. Call the model (or mock)
    if use_api:
        try:
            answer = ask_anthropic(task, system)
        except Exception as exc:
            err_id = mem.log_error(MODEL, task_type, error=exc)
            mem.resolve(err_id, f"API error: {exc}")
            mem.report_outcome(inj, success=False)
            raise
    else:
        answer = mock_response

    # 3. Evaluate quality (simple heuristic -- real agents use LLM-as-judge)
    success = len(answer) > 10 and "i don't know" not in answer.lower()
    mem.report_outcome(inj, success=success)

    return answer


# -- Offline demo ------------------------------------------------------------

def _demo() -> None:
    """Self-contained offline demo -- no API key needed.

    Simulates two rounds: first fails (bad analysis), second succeeds
    after errlore injects the lesson from the first failure.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mem = AgentMemory(Path(tmp) / "agent_memory")

        # Round 1: agent produces a bad analysis
        err_id = mem.log_error(
            MODEL, "analysis",
            error="Confused fiscal year (FY) with calendar year, "
                  "leading to off-by-one-quarter comparison",
        )
        mem.resolve(
            err_id,
            "Explicitly map FY quarters to calendar dates before comparing",
            lesson="When analyzing financial reports, always identify whether "
                   "dates refer to fiscal year or calendar year, and convert "
                   "to a common reference before making comparisons.",
        )

        # Round 2: same domain -- errlore auto-injects the lesson
        task2 = "Compare Q3 revenue across the last three fiscal years."
        answer = run_with_errlore(
            mem, task2,
            task_type="analysis",
            mock_response=(
                "FY2024-Q3 (Jul-Sep 2023): $12.4M\n"
                "FY2025-Q3 (Jul-Sep 2024): $14.1M (+13.7%)\n"
                "FY2026-Q3 (Jul-Sep 2025): $15.8M (+12.1%)\n"
                "Note: FY starts October 1, so FY Q3 = calendar Jul-Sep."
            ),
        )

        # Show injection
        inj = mem.inject_for(task2, model=MODEL, task_type="analysis")
        print("=== errlore injection ===")
        print(inj.text or "(empty)")
        print()
        print(f"=== Agent answer (mock) ===\n{answer}")
        print()

        # Trust: which model does errlore trust most?
        best = mem.best_model()
        print(f"=== Best model (general domain) === {best or 'none registered'}")
        print()

        stats = mem.stats()
        print("=== Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    _demo()
