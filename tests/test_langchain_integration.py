"""LangChain integration: callback capture + middleware injection loop.

Skipped entirely when langchain is not installed (it is in the dev extra).
The middleware test runs a REAL create_agent loop against a fake chat model
that records the exact messages it receives -- so "the lesson reached the
prompt" is asserted end-to-end, not against internals.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from errlore import AgentMemory
from errlore.integrations.langchain import ErrloreCallbackHandler


class TestCallbackHandler:
    def test_tool_error_is_captured(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        handler = ErrloreCallbackHandler(mem, model="gpt-test")
        handler.on_tool_error(ValueError("boom in tool"), run_id=uuid4())
        stats = mem.stats()
        assert stats["errors_total"] == 1
        raw = (data_dir / "errors.jsonl").read_text()
        assert "boom in tool" in raw
        assert "ValueError" in raw

    def test_llm_error_is_captured(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        handler = ErrloreCallbackHandler(mem, model="gpt-test")
        handler.on_llm_error(TimeoutError("llm timed out"), run_id=uuid4())
        assert mem.stats()["errors_total"] == 1

    def test_chain_errors_off_by_default(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        handler = ErrloreCallbackHandler(mem, model="gpt-test")
        handler.on_chain_error(RuntimeError("chain fail"), run_id=uuid4())
        assert mem.stats()["errors_total"] == 0
        opt_in = ErrloreCallbackHandler(
            mem, model="gpt-test", capture_chain_errors=True
        )
        opt_in.on_chain_error(RuntimeError("chain fail"), run_id=uuid4())
        assert mem.stats()["errors_total"] == 1

    def test_capture_failure_never_raises(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        handler = ErrloreCallbackHandler(mem, model="gpt-test")
        mem.log_error = None  # type: ignore[assignment]  # sabotage
        handler.on_tool_error(ValueError("x"), run_id=uuid4())  # must not raise


# ---------------------------------------------------------------------------
# Middleware: real create_agent loop with a recording fake model
# ---------------------------------------------------------------------------

pytest.importorskip("langchain")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402

from errlore.integrations.langchain import ErrloreMiddleware  # noqa: E402

_seen_prompts: list[list[BaseMessage]] = []


class _RecordingModel(GenericFakeChatModel):
    """Fake chat model that records every prompt it is asked to complete."""

    def _generate(self, messages: list[BaseMessage], *args: object, **kwargs: object):  # type: ignore[no-untyped-def,override]
        _seen_prompts.append(list(messages))
        return super()._generate(messages, *args, **kwargs)  # type: ignore[arg-type]


def _seed_lesson(mem: AgentMemory) -> None:
    eid = mem.log_error("m", "agent", "DateError: hallucinated invoice year")
    mem.resolve(
        eid, "fixed", lesson="when extracting invoice dates verify the year digits"
    )


class TestMiddleware:
    def _run_agent(self, mem: AgentMemory) -> ErrloreMiddleware:
        mw = ErrloreMiddleware(mem, model="fake-model", task_type="agent")
        agent = create_agent(
            model=_RecordingModel(messages=iter([AIMessage(content="done")])),
            tools=[],
            middleware=[mw],
        )
        agent.invoke(
            {"messages": [("user", "extract the invoice dates from this PDF")]}
        )
        return mw

    def test_lesson_reaches_model_prompt(self, data_dir: Path) -> None:
        _seen_prompts.clear()
        mem = AgentMemory(data_dir, trust=False)
        _seed_lesson(mem)
        mw = self._run_agent(mem)

        assert _seen_prompts, "fake model was never called"
        rendered = "\n".join(str(m.content) for m in _seen_prompts[0])
        assert "[LESSONS FROM PAST FAILURES]" in rendered
        assert "verify the year digits" in rendered
        assert mw.last_injection is not None
        assert mw.last_injection.lesson_ids

    def test_report_closes_the_loop(self, data_dir: Path) -> None:
        _seen_prompts.clear()
        mem = AgentMemory(data_dir, trust=False)
        _seed_lesson(mem)
        mw = self._run_agent(mem)

        assert mw.report(success=True) is True
        assert mw.report(success=True) is False  # idempotent
        lessons = mem.lessons()
        assert lessons[0].applied_count == 1
        # The reported marker is durable.
        rows = [
            json.loads(line)
            for line in (data_dir / "injections.jsonl").read_text().splitlines()
        ]
        assert any(r.get("event") == "reported" for r in rows)

    def test_no_lessons_means_untouched_prompt(self, data_dir: Path) -> None:
        _seen_prompts.clear()
        mem = AgentMemory(data_dir, trust=False)  # empty memory
        mw = self._run_agent(mem)
        rendered = "\n".join(str(m.content) for m in _seen_prompts[0])
        assert "[LESSONS FROM PAST FAILURES]" not in rendered
        # Handle still issued (empty injection is reportable).
        assert mw.last_injection is not None

    def test_report_without_run_returns_false(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        mw = ErrloreMiddleware(mem)
        assert mw.report(success=True) is False
