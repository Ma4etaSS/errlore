"""
title: errlore Feedback (close the loop)
author: Ma4etaSS
author_url: https://github.com/Ma4etaSS/errlore
version: 0.1.0
required_open_webui_version: 0.5.0
requirements: errlore
"""

# Companion Action for the "errlore Memory" Filter.
#
# Adds a button under each assistant message.  Clicking it asks whether the
# response was good; the answer closes errlore's reinforcement loop:
#   good     -> report_outcome(success=True)   (lessons reinforced, trust up)
#   bad      -> report_outcome(success=False)  + optional lesson capture:
#               the error is logged and, if the user types a takeaway,
#               resolved into a new lesson for future injections.

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from errlore import AgentMemory


class Action:
    class Valves(BaseModel):
        data_dir: str = "/app/backend/data/errlore"
        task_type: str = "chat"

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._mem: AgentMemory | None = None
        self._mem_dir: str | None = None

    def _memory(self) -> AgentMemory:
        if self._mem is None or self._mem_dir != self.valves.data_dir:
            self._mem = AgentMemory(self.valves.data_dir)
            self._mem_dir = self.valves.data_dir
        return self._mem

    def _handle_for(self, chat_id: str) -> str | None:
        path = Path(self.valves.data_dir) / "owui_chat_handles.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text()).get(chat_id)
        except (OSError, json.JSONDecodeError):
            return None

    async def action(
        self,
        body: dict,
        __user__: dict | None = None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> dict | None:
        if __event_call__ is None:
            return None

        mem = self._memory()
        chat_id = str(body.get("chat_id") or "")
        model = str(body.get("model", "unknown"))

        good = await __event_call__(
            {
                "type": "confirmation",
                "data": {
                    "title": "errlore feedback",
                    "message": "Was this response good (no repeated mistakes)?",
                },
            }
        )

        handle_id = self._handle_for(chat_id) if chat_id else None
        reported = False
        if handle_id:
            try:
                reported = mem.report_outcome(handle_id, bool(good))
            except KeyError:
                reported = False

        if good:
            status = "reinforced" if reported else "no pending injection"
            if __event_emitter__ is not None:
                await __event_emitter__(
                    {"type": "status", "data": {"description": f"errlore: {status}"}}
                )
            return None

        # Bad response: log the error and offer to capture a lesson.
        content = str(body.get("content", ""))[:200]
        err_id = mem.log_error(model, self.valves.task_type, f"bad_response: {content}")
        lesson = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "Capture a lesson (optional)",
                    "message": (
                        "What should the assistant do differently next time? "
                        "Leave empty to skip."
                    ),
                    "placeholder": "e.g. Always cite the source when quoting numbers",
                },
            }
        )
        if isinstance(lesson, str) and lesson.strip():
            mem.resolve(err_id, "user feedback", lesson=lesson.strip())
            desc = "errlore: lesson saved"
        else:
            desc = "errlore: error logged"
        if __event_emitter__ is not None:
            await __event_emitter__({"type": "status", "data": {"description": desc}})
        return None
