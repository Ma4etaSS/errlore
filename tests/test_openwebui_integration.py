"""Tests for the Open WebUI integration (Filter + Action).

The integration files live outside the package (integrations/openwebui/) and
carry OWUI frontmatter docstrings, but they are plain Python -- load them via
importlib and exercise the full loop without any Open WebUI server.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from errlore import AgentMemory

INTEGRATIONS = Path(__file__).parent.parent / "integrations" / "openwebui"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, INTEGRATIONS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def owui(data_dir: Path):
    """Filter + Action wired to one tmp data dir, with a seeded lesson."""
    filt = _load("errlore_memory_filter").Filter()
    act = _load("errlore_feedback_action").Action()
    filt.valves.data_dir = str(data_dir)
    act.valves.data_dir = str(data_dir)

    mem = AgentMemory(data_dir)
    err = mem.log_error("gpt-5.5", "chat", "SomeError: hallucinated dates in table")
    mem.resolve(err, "fixed", lesson="for tables with dates demand ISO-8601 format")
    return filt, act, data_dir


def test_inlet_injects_lessons_and_stores_handle(owui) -> None:
    filt, _act, data_dir = owui
    body = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "extract dates from this table"}],
    }
    out = asyncio.run(filt.inlet(body, __metadata__={"chat_id": "chat-1"}))

    assert out["messages"][0]["role"] == "system"
    assert "LESSONS FROM PAST FAILURES" in out["messages"][0]["content"]
    handles = json.loads((data_dir / "owui_chat_handles.json").read_text())
    assert handles.get("chat-1")


def test_inlet_without_user_message_is_noop(owui) -> None:
    filt, _act, _ = owui
    body = {"model": "m", "messages": [{"role": "assistant", "content": "hi"}]}
    out = asyncio.run(filt.inlet(body))
    assert all(m["role"] != "system" for m in out["messages"])


def test_action_good_feedback_reinforces(owui) -> None:
    filt, act, data_dir = owui
    body = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "extract dates from this table"}],
    }
    asyncio.run(filt.inlet(body, __metadata__={"chat_id": "chat-2"}))

    async def yes(_event):
        return True

    asyncio.run(
        act.action({"chat_id": "chat-2", "model": "gpt-5.5"}, __event_call__=yes)
    )
    lesson = json.loads((data_dir / "lessons.jsonl").read_text().splitlines()[0])
    assert lesson["applied_count"] == 1
    # Second click on the same chat handle: idempotent, no double reinforce.
    asyncio.run(
        act.action({"chat_id": "chat-2", "model": "gpt-5.5"}, __event_call__=yes)
    )
    lesson = json.loads((data_dir / "lessons.jsonl").read_text().splitlines()[0])
    assert lesson["applied_count"] == 1


def test_action_bad_feedback_captures_lesson(owui) -> None:
    filt, act, data_dir = owui
    body = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "summarize the contract"}],
    }
    asyncio.run(filt.inlet(body, __metadata__={"chat_id": "chat-3"}))

    calls = iter([False, "always quote the termination clause verbatim"])

    async def event_call(_event):
        return next(calls)

    asyncio.run(
        act.action(
            {"chat_id": "chat-3", "model": "gpt-5.5", "content": "bad summary..."},
            __event_call__=event_call,
        )
    )
    mem = AgentMemory(data_dir)
    patterns = [le.solution for le in mem.lessons()]
    assert any("termination clause" in s for s in patterns)
    stats = mem.stats()
    assert stats["errors_total"] >= 2  # seeded + bad_response
