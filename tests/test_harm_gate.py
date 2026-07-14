"""Integration tests: the harm gate wired through AgentMemory.

Drives the real report_outcome loop and asserts a lesson that keeps failing
stops being injected, while healthy lessons keep flowing.
"""

from __future__ import annotations

from pathlib import Path

from errlore import AgentMemory


def _fail_lesson(mem: AgentMemory, task: str, task_type: str, times: int) -> None:
    """Inject for *task* and report failure *times* times."""
    for _ in range(times):
        inj = mem.inject_for(task, model="m", task_type=task_type)
        mem.report_outcome(inj, success=False)


def _succeed_lesson(mem: AgentMemory, task: str, task_type: str, times: int) -> None:
    for _ in range(times):
        inj = mem.inject_for(task, model="m", task_type=task_type)
        mem.report_outcome(inj, success=True)


def test_failing_lesson_gets_quarantined_out_of_injection(data_dir: Path) -> None:
    mem = AgentMemory(data_dir, trust=False)
    mem.add_lesson(
        "extract dates from contracts",
        "demand ISO-8601",
        task_type="extraction",
    )

    # Before any bad outcomes the lesson injects.
    inj = mem.inject_for("extract dates from contracts", "m", task_type="extraction")
    assert "ISO-8601" in inj.text

    # Drive failure cycles. The gate is self-limiting: once the harm posterior
    # crosses the bar the lesson stops being injected, so it also stops
    # accruing failures -- it caps the damage at a handful of bad injections.
    _fail_lesson(mem, "extract dates from contracts", "extraction", times=8)

    assert mem.stats()["lessons_quarantined"] == 1
    q = mem.quarantined_lessons()
    assert len(q) == 1
    # Crossed the bar after ~4 harmful injections, then froze (did not reach 8).
    assert 4 <= q[0].failure_count <= 5

    # The lesson is now withheld from injection.
    inj2 = mem.inject_for("extract dates from contracts", "m", task_type="extraction")
    assert "ISO-8601" not in inj2.text


def test_harm_gate_disabled_keeps_injecting(data_dir: Path) -> None:
    mem = AgentMemory(data_dir, trust=False, harm_gate=False)
    mem.add_lesson(
        "extract dates from contracts",
        "demand ISO-8601",
        task_type="extraction",
    )
    _fail_lesson(mem, "extract dates from contracts", "extraction", times=6)

    # Still quarantined by the verdict...
    assert mem.stats()["lessons_quarantined"] == 1
    # ...but with the gate off it is injected anyway.
    inj = mem.inject_for("extract dates from contracts", "m", task_type="extraction")
    assert "ISO-8601" in inj.text


def test_healthy_lesson_is_not_gated(data_dir: Path) -> None:
    mem = AgentMemory(data_dir, trust=False)
    mem.add_lesson(
        "extract dates from contracts",
        "demand ISO-8601",
        task_type="extraction",
    )
    _succeed_lesson(mem, "extract dates from contracts", "extraction", times=20)

    assert mem.stats()["lessons_quarantined"] == 0
    inj = mem.inject_for("extract dates from contracts", "m", task_type="extraction")
    assert "ISO-8601" in inj.text


def test_counters_persist_across_reload(data_dir: Path) -> None:
    mem = AgentMemory(data_dir, trust=False)
    mem.add_lesson("some pattern", "some fix", task_type="t")
    _fail_lesson(mem, "some pattern", "t", times=3)
    _succeed_lesson(mem, "some pattern", "t", times=2)

    reloaded = AgentMemory(data_dir, trust=False)
    lesson = reloaded.lessons()[0]
    assert lesson.failure_count == 3
    assert lesson.success_count == 2
