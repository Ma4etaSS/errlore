#!/usr/bin/env python3
"""LangChain integration example -- the closed failure-memory loop.

Uses the first-class integration (``errlore.integrations.langchain``):

* ``ErrloreMiddleware`` injects relevant lessons into the agent's system
  prompt on each run (langchain >= 1.0 ``create_agent`` middleware), and
* ``ErrloreCallbackHandler`` auto-captures tool/LLM errors into memory.

Requires: pip install errlore[langchain]

Run offline (no API key needed) -- the demo uses a fake chat model:
    python examples/langchain_agent.py

Swap ``_demo_model()`` for ``"gpt-5.5"`` / any chat model to run for real.
"""

from __future__ import annotations

import tempfile

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from errlore import AgentMemory
from errlore.integrations.langchain import (
    ErrloreCallbackHandler,
    ErrloreMiddleware,
)


def _demo_model() -> GenericFakeChatModel:
    """Offline stand-in; replace with a real chat model string or instance."""
    return GenericFakeChatModel(
        messages=iter([AIMessage(content="Dates: 2026-05-01, 2026-06-01.")])
    )


def main() -> None:
    data_dir = tempfile.mkdtemp(prefix="errlore-langchain-")
    mem = AgentMemory(data_dir, trust=False)

    # Monday: a failure gets resolved into a lesson.
    err_id = mem.log_error(
        "demo-model", "agent", "DateError: read the invoice year as 2025"
    )
    mem.resolve(
        err_id,
        "fixed by re-checking the year digits",
        lesson="when extracting invoice dates, verify the year digits "
        "against the document header",
    )

    # Tuesday: the same class of task -- the lesson rides along automatically.
    mw = ErrloreMiddleware(mem, model="demo-model", task_type="agent")
    agent = create_agent(
        model=_demo_model(),
        tools=[],
        middleware=[mw],
    )
    result = agent.invoke(
        {"messages": [("user", "extract the invoice dates from this PDF")]},
        config={"callbacks": [ErrloreCallbackHandler(mem, model="demo-model")]},
    )

    print("agent answer:", result["messages"][-1].content)
    assert mw.last_injection is not None
    print("\ninjected block was:\n" + (mw.last_injection.text or "(empty)"))

    # Close the loop AFTER validating the result with your own check
    # (schema, tests, exit code -- never the model's own opinion).
    mw.report(success=True)
    print("\nlesson reinforced; stats:", mem.stats())


if __name__ == "__main__":
    main()
