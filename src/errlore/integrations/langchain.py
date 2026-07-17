"""LangChain integration -- failure memory for agent loops.

Two pieces; use either or both:

* :class:`ErrloreCallbackHandler` -- auto-captures tool/LLM errors into
  :class:`~errlore.facade.AgentMemory` (the same capture loop as the Claude
  Code hooks). Works with anything that accepts langchain callbacks --
  ``create_agent``, LCEL chains, plain chat models.
* :class:`ErrloreMiddleware` -- for langchain >= 1.0 ``create_agent``:
  injects relevant lessons into the system prompt on the first model call of
  each run, and exposes the injection handle so the outcome can be reported
  back (:meth:`ErrloreMiddleware.report`) to close the reinforcement loop.

Install with the extra::

    pip install errlore[langchain]

The callback handler alone needs only ``langchain-core``; the middleware
needs the full ``langchain`` package (``create_agent``).

Quickstart::

    from errlore import AgentMemory
    from errlore.integrations.langchain import (
        ErrloreCallbackHandler,
        ErrloreMiddleware,
    )
    from langchain.agents import create_agent

    mem = AgentMemory("./errlore-data")
    mw = ErrloreMiddleware(mem, model="gpt-5.5", task_type="agent")

    agent = create_agent(model="gpt-5.5", tools=[...], middleware=[mw])
    result = agent.invoke(
        {"messages": [("user", "extract the invoice dates")]},
        config={"callbacks": [ErrloreCallbackHandler(mem, model="gpt-5.5")]},
    )

    # After validating the result with YOUR check (schema, tests, exit code):
    mw.report(success=True)
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from errlore.facade import AgentMemory, Injection

logger = logging.getLogger("errlore.integrations.langchain")

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
except ImportError as _e:  # pragma: no cover - exercised only without extras
    raise ImportError(
        "errlore.integrations.langchain requires langchain-core. "
        "Install with: pip install errlore[langchain]"
    ) from _e

_MAX_ERROR_LEN = 300


class ErrloreCallbackHandler(BaseCallbackHandler):
    """Auto-capture tool/LLM failures into an :class:`AgentMemory`.

    Pass it in ``config={"callbacks": [...]}`` (or any ``callbacks=`` slot).
    Every ``on_tool_error`` / ``on_llm_error`` becomes an errlore error
    record, feeding the known-issues warnings and (once you resolve errors
    into lessons) the lesson loop.

    Chain errors are NOT captured by default: a failing tool inside an agent
    also fails its enclosing chains, so capturing both double-counts one
    failure. Set ``capture_chain_errors=True`` for chain-only pipelines.

    Defensive by contract (like the Claude Code hooks): capture failures are
    swallowed with a log warning, never raised into the agent loop.

    Args:
        memory: The AgentMemory to record into.
        model: Model name recorded with each error (used for per-model
            weakness tracking). Use the real model id when you have it.
        capture_chain_errors: Also record ``on_chain_error`` events.
    """

    def __init__(
        self,
        memory: AgentMemory,
        *,
        model: str = "langchain",
        capture_chain_errors: bool = False,
    ) -> None:
        self._mem = memory
        self._model = model
        self._capture_chain = capture_chain_errors

    def _capture(self, task_type: str, error: BaseException) -> None:
        try:
            self._mem.log_error(
                self._model,
                task_type,
                f"{type(error).__name__}: {str(error)[:_MAX_ERROR_LEN]}",
            )
        except Exception:  # never break the agent loop
            logger.warning("errlore capture failed", exc_info=True)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._capture("tool", error)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        self._capture("llm", error)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        if self._capture_chain:
            self._capture("chain", error)


def _text_of(content: object) -> str:
    """Best-effort plain text of a message content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(p for p in parts if p)
    return str(content)


try:
    from langchain.agents.middleware import (
        AgentMiddleware,
        ModelRequest,
        ModelResponse,
    )

    _HAS_AGENT_MIDDLEWARE = True
except ImportError:  # pragma: no cover - exercised only without langchain
    _HAS_AGENT_MIDDLEWARE = False

if _HAS_AGENT_MIDDLEWARE:

    class ErrloreMiddleware(AgentMiddleware):
        """Inject relevant lessons into the agent's system prompt (closed loop).

        On the FIRST model call of each agent run (detected as "no AI message
        in the transcript yet"), searches lessons relevant to the user's task
        and appends the errlore block to the system message. Subsequent model
        calls in the same run re-append the same block, so the prompt stays
        consistent through the tool loop without issuing new handles.

        After the run, validate the result with YOUR check and call
        :meth:`report` -- lessons that helped are reinforced, lessons that
        didn't decay, and the harm gate learns which lessons hurt.

        One middleware instance tracks one agent invocation at a time; give
        concurrent runners their own instance (they can share the data_dir --
        errlore's stores are cross-process safe).

        Args:
            memory: The AgentMemory to search and report into.
            model: Model name for known-issue lookup and trust reporting.
            task_type: errlore task category for narrower lesson search.
            domain: Trust domain (default ``"general"``).
        """

        def __init__(
            self,
            memory: AgentMemory,
            *,
            model: str = "langchain-agent",
            task_type: str = "agent",
            domain: str | None = None,
        ) -> None:
            super().__init__()
            self._mem = memory
            self._model = model
            self._task_type = task_type
            self._domain = domain
            #: Handle of the current/most recent run's injection.
            self.last_injection: Injection | None = None
            self._block_text: str = ""

        # -- core logic (shared by sync and async wrappers) ---------------

        def _prepare(self, request: ModelRequest) -> ModelRequest:
            first_call = not any(
                isinstance(m, AIMessage) for m in request.messages
            )
            if first_call:
                task = self._task_from(request)
                inj = self._mem.inject_for(
                    task,
                    model=self._model,
                    task_type=self._task_type,
                    domain=self._domain,
                )
                self.last_injection = inj
                self._block_text = inj.text
            if not self._block_text:
                return request
            return self._append_system(request, self._block_text)

        def _task_from(self, request: ModelRequest) -> str:
            for m in reversed(request.messages):
                if isinstance(m, HumanMessage):
                    return _text_of(m.content)
            return ""

        @staticmethod
        def _append_system(request: ModelRequest, text: str) -> ModelRequest:
            sysmsg = request.system_message
            blocks: list[Any] = (
                list(sysmsg.content_blocks) if sysmsg is not None else []
            )
            blocks.append({"type": "text", "text": text})
            return request.override(system_message=SystemMessage(content=blocks))

        # -- middleware hooks ---------------------------------------------

        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            return handler(self._prepare(request))

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await handler(self._prepare(request))

        # -- reinforcement ------------------------------------------------

        def report(self, success: bool, *, outcome: float | None = None) -> bool:
            """Close the loop for the most recent run's injection.

            Call this after validating the agent's result with your own
            deterministic check. Returns ``False`` when there is nothing to
            report (no run yet, or already reported).
            """
            if self.last_injection is None:
                return False
            return self._mem.report_outcome(
                self.last_injection, success, outcome=outcome
            )

else:  # pragma: no cover - exercised only without langchain installed

    def __getattr__(name: str) -> Any:
        if name == "ErrloreMiddleware":
            raise ImportError(
                "ErrloreMiddleware requires the full langchain package "
                "(create_agent). Install with: pip install errlore[langchain]"
            )
        raise AttributeError(name)
