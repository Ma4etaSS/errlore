#!/usr/bin/env python3
"""LangChain integration example -- errlore as system-prompt middleware.

errlore enriches the system message before every LLM call, injecting
relevant lessons and known-issue warnings.  This is the ideal integration
point: no framework monkey-patching, works with any LangChain chat model.

Requires: pip install langchain-openai  (or langchain-anthropic)

Note: LangChain >=1.x does NOT expose ``create_agent`` / ``dynamic_prompt``
middleware.  This example uses ``ChatOpenAI`` + ``SystemMessage`` directly,
which is the stable, documented integration pattern.

Run offline (no API key needed):
    python examples/langchain_agent.py

The ``if __name__`` block uses a mock instead of a real LLM call.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from errlore import AgentMemory, Injection

# Model is just a label to errlore (it never calls the API itself). Override
# with your own; the default is a small, cheap, widely-available option.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# -- errlore + LangChain glue ------------------------------------------------

def build_system_message(
    mem: AgentMemory,
    task: str,
    *,
    task_type: str = "code_generation",
) -> tuple[str, Injection]:
    """Build a system message enriched with errlore context.

    Returns the system text and the Injection handle (needed for
    ``report_outcome`` later).
    """
    inj = mem.inject_for(task, model=MODEL, task_type=task_type)
    base = "You are a helpful coding assistant. Write clean, tested code."
    system = base + "\n\n" + inj.text if inj.text else base
    return system, inj


def ask_langchain(task: str, system: str) -> str:
    """Call LangChain ChatOpenAI with a system + user message.

    Requires OPENAI_API_KEY in the environment.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=task),
    ])
    return str(response.content)


def run_with_errlore(
    mem: AgentMemory,
    task: str,
    *,
    use_api: bool = False,
    mock_response: str = "",
) -> str:
    """Run a task through errlore-augmented LangChain agent.

    When *use_api* is False (default), returns *mock_response* instead
    of calling LangChain/OpenAI -- useful for offline demos and tests.
    """
    system, inj = build_system_message(mem, task)

    if use_api:
        try:
            answer = ask_langchain(task, system)
        except Exception as exc:
            err_id = mem.log_error(MODEL, "code_generation", error=exc)
            mem.resolve(err_id, f"LangChain call failed: {exc}")
            mem.report_outcome(inj, success=False)
            raise
    else:
        answer = mock_response

    # Evaluate and close the loop (simple heuristic -- real agents use LLM-as-judge)
    success = len(answer) > 0
    mem.report_outcome(inj, success=success)

    return answer


# -- Offline demo ------------------------------------------------------------

def _demo() -> None:
    """Self-contained offline demo -- no API key needed.

    Shows how errlore accumulates knowledge across multiple agent runs
    and enriches prompts with relevant lessons.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mem = AgentMemory(Path(tmp) / "agent_memory")

        # --- Seed two past failures ---

        # Failure 1: agent forgot error handling
        err1 = mem.log_error(
            MODEL, "code_generation",
            error="Generated function has no try/except; crashes on bad input",
        )
        mem.resolve(
            err1,
            "Wrapped parsing in try/except with descriptive ValueError",
            lesson="Always wrap I/O and parsing code in try/except blocks "
                   "and raise descriptive errors instead of letting raw "
                   "exceptions propagate.",
        )

        # Failure 2: agent used deprecated API
        err2 = mem.log_error(
            MODEL, "code_generation",
            error="Used requests.get without timeout; hangs in production",
        )
        mem.resolve(
            err2,
            "Added timeout=30 to all requests calls",
            lesson="Always pass an explicit timeout to HTTP calls "
                   "(requests, httpx, aiohttp). Default no-timeout hangs "
                   "in production.",
        )

        # --- Run a new task -- errlore injects both lessons ---

        task = "Write a Python function that fetches JSON from a URL and parses it."
        answer = run_with_errlore(
            mem, task,
            mock_response=(
                "def fetch_json(url: str, timeout: int = 30) -> dict:\n"
                "    import requests\n"
                "    try:\n"
                "        resp = requests.get(url, timeout=timeout)\n"
                "        resp.raise_for_status()\n"
                "        return resp.json()\n"
                "    except requests.RequestException as exc:\n"
                "        raise ValueError(f'Failed to fetch {url}: {exc}') from exc\n"
            ),
        )

        # Show what errlore injected
        system, _inj = build_system_message(mem, task)
        print("=== System message (with errlore injection) ===")
        print(system)
        print()
        print(f"=== Agent answer (mock) ===\n{answer}")
        print()

        stats = mem.stats()
        print("=== Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    _demo()
