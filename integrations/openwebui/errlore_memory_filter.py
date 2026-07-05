"""
title: errlore Memory (lessons from failures)
author: Ma4etaSS
author_url: https://github.com/Ma4etaSS/errlore
version: 0.1.0
required_open_webui_version: 0.5.0
requirements: errlore
"""

# errlore Filter for Open WebUI.
#
# inlet:  looks up lessons + per-model KNOWN ISSUES for the user's message
#         and injects them as a system message.  The injection handle is
#         persisted (chat_id -> handle_id) so the companion Action
#         ("errlore Feedback") can close the reinforcement loop later.
# outlet: pass-through (outcome reporting is explicit via the Action --
#         errlore never fakes reinforcement it cannot observe).

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from errlore import AgentMemory


def _load_handles(path: Path) -> dict[str, str]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_handles(path: Path, handles: dict[str, str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(handles))
    tmp.replace(path)


class Filter:
    class Valves(BaseModel):
        data_dir: str = "/app/backend/data/errlore"
        max_lessons: int = 3
        task_type: str = "chat"
        embeddings: bool = False  # requires errlore[embeddings] in requirements

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._mem: AgentMemory | None = None
        self._mem_dir: str | None = None

    def _memory(self) -> AgentMemory:
        # Re-create memory if the valve changed.
        if self._mem is None or self._mem_dir != self.valves.data_dir:
            self._mem = AgentMemory(
                self.valves.data_dir,
                max_lessons=self.valves.max_lessons,
                embeddings=self.valves.embeddings,
            )
            self._mem_dir = self.valves.data_dir
        return self._mem

    def _handles_path(self) -> Path:
        return Path(self.valves.data_dir) / "owui_chat_handles.json"

    async def inlet(
        self,
        body: dict,
        __user__: dict | None = None,
        __metadata__: dict | None = None,
    ) -> dict:
        messages: list[dict[str, Any]] = body.setdefault("messages", [])
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if last_user is None or not isinstance(last_user.get("content"), str):
            return body

        mem = self._memory()
        model = str(body.get("model", "unknown"))
        inj = mem.inject_for(
            last_user["content"], model=model, task_type=self.valves.task_type
        )
        if inj.text:
            messages.insert(0, {"role": "system", "content": inj.text})

        chat_id = str((__metadata__ or {}).get("chat_id") or body.get("chat_id") or "")
        if chat_id:
            path = self._handles_path()
            handles = _load_handles(path)
            handles[chat_id] = inj.handle_id
            # Keep the map bounded.
            if len(handles) > 500:
                for key in list(handles)[:-500]:
                    handles.pop(key, None)
            _save_handles(path, handles)
        return body

    async def outlet(self, body: dict, __user__: dict | None = None) -> dict:
        return body
