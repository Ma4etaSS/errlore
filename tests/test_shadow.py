"""Tests for shadow mode: counterfactual queue + graduation loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from errlore import AgentMemory
from errlore.shadow import CounterfactualQueue


class TestCounterfactualQueue:
    def test_enqueue_then_pending(self, data_dir: Path) -> None:
        q = CounterfactualQueue(data_dir)
        cf_id = q.enqueue(["l1"], "m", "base prompt", "base prompt\n[LESSON]")
        pend = q.pending()
        assert len(pend) == 1
        assert pend[0].cf_id == cf_id
        assert pend[0].lesson_ids == ["l1"]

    def test_resolve_clears_pending_and_returns_lessons(self, data_dir: Path) -> None:
        q = CounterfactualQueue(data_dir)
        cf_id = q.enqueue(["l1", "l2"], "m", "b", "i")
        lids = q.resolve(cf_id, baseline_passed=True, injected_passed=False)
        assert lids == ["l1", "l2"]
        assert q.pending() == []

    def test_resolve_unknown_raises(self, data_dir: Path) -> None:
        q = CounterfactualQueue(data_dir)
        with pytest.raises(KeyError):
            q.resolve("nope", True, True)

    def test_resolve_duplicate_returns_none(self, data_dir: Path) -> None:
        q = CounterfactualQueue(data_dir)
        cf_id = q.enqueue(["l1"], "m", "b", "i")
        assert q.resolve(cf_id, True, True) == ["l1"]
        assert q.resolve(cf_id, True, True) is None

    def test_survives_reload(self, data_dir: Path) -> None:
        q1 = CounterfactualQueue(data_dir)
        cf_id = q1.enqueue(["l1"], "m", "b", "i")
        q2 = CounterfactualQueue(data_dir)
        assert [c.cf_id for c in q2.pending()] == [cf_id]

    def test_compaction_drops_closed_queued_keeps_pending(
        self, data_dir: Path
    ) -> None:
        """Past the threshold, resolved trials lose their heavy queued record
        (prompts), pending trials survive intact, and a duplicate resolve on a
        compacted trial still returns None (not KeyError)."""
        q = CounterfactualQueue(data_dir)
        q._COMPACT_THRESHOLD = 4  # force compaction quickly

        closed = [q.enqueue(["l1"], "m", f"base{i}", f"inj{i}") for i in range(2)]
        pending_id = q.enqueue(["l2"], "m", "base-pending", "inj-pending")
        for cf in closed:
            assert q.resolve(cf, True, True) == ["l1"]

        import json

        rows = [
            json.loads(line)
            for line in (data_dir / "counterfactuals.jsonl").read_text().splitlines()
        ]
        queued_ids = {r["cf_id"] for r in rows if r["event"] == "queued"}
        resolved_ids = {r["cf_id"] for r in rows if r["event"] == "resolved"}
        for cf in closed:
            assert cf not in queued_ids  # heavy record dropped
            assert cf in resolved_ids  # marker kept
        assert pending_id in queued_ids

        # Idempotency survives compaction.
        assert q.resolve(closed[0], True, True) is None
        # Pending trial is still discoverable and resolvable.
        assert [c.cf_id for c in q.pending()] == [pending_id]
        assert q.resolve(pending_id, True, False) == ["l2"]

    def test_resolve_is_idempotent_across_separate_instances(
        self, data_dir: Path
    ) -> None:
        """Two queues (≈two processes) resolving the same cf_id: only one wins.

        Guards the fix where resolve() must read fresh under the lock -- a stale
        per-instance cache could miss the other's 'resolved' marker and
        double-resolve, corrupting the graduation posterior.
        """
        writer = CounterfactualQueue(data_dir)
        cf_id = writer.enqueue(["l1"], "m", "b", "i")

        q1 = CounterfactualQueue(data_dir)
        q2 = CounterfactualQueue(data_dir)
        # Warm each instance's read cache before either resolves.
        assert q1.pending() and q2.pending()

        r1 = q1.resolve(cf_id, True, True)
        r2 = q2.resolve(cf_id, True, True)
        assert {bool(r1), bool(r2)} == {True, False}
        assert CounterfactualQueue(data_dir).pending() == []


class TestCounterSemantics:
    def test_trial_types_map_to_right_counters(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        lid = mem.add_lesson("p", "s", task_type="t")
        store = mem._store  # type: ignore[attr-defined]
        # baseline passed, injection broke -> harm_break
        store.record_counterfactual(lid, baseline_passed=True, injected_passed=False)
        # baseline passed, injection kept -> harm_keep
        store.record_counterfactual(lid, baseline_passed=True, injected_passed=True)
        # baseline failed, injection fixed -> fix_yes
        store.record_counterfactual(lid, baseline_passed=False, injected_passed=True)
        # baseline failed, injection did not -> fix_no
        store.record_counterfactual(lid, baseline_passed=False, injected_passed=False)
        le = mem.lessons()[0]
        assert (le.harm_break, le.harm_keep, le.fix_yes, le.fix_no) == (1, 1, 1, 1)


class TestGraduationThroughFacade:
    def _shadow_trial(
        self, mem: AgentMemory, task: str, tt: str, base_pass: bool, inj_pass: bool
    ) -> None:
        inj = mem.inject_for(task, "m", task_type=tt, mode="shadow")
        cf_id = mem.enqueue_counterfactual(inj, baseline_prompt="do the task")
        assert mem.report_counterfactual_outcome(cf_id, base_pass, inj_pass) is True

    def test_clean_and_useful_lesson_promotes(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        lid = mem.add_lesson("format dates as ISO", "use ISO-8601", task_type="extraction")

        # 60 clean success-trials (injection never breaks a pass) -> safe.
        for _ in range(60):
            self._shadow_trial(mem, "format dates as ISO", "extraction", True, True)
        # Not useful yet (no observed fix) -> still holds.
        assert mem.graduation_status(lid) == "hold"

        # One failure-trial where injection fixes it -> crosses usefulness.
        self._shadow_trial(mem, "format dates as ISO", "extraction", False, True)
        assert mem.graduation_status(lid) == "promote"
        assert lid in [le.id for le in mem.graduated_lessons()]

    def test_harmful_lesson_quarantines(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        lid = mem.add_lesson("risky advice", "do X", task_type="t")
        # 5 harms in 20 success-trials -> quarantine.
        for _ in range(5):
            self._shadow_trial(mem, "risky advice", "t", True, False)
        for _ in range(15):
            self._shadow_trial(mem, "risky advice", "t", True, True)
        assert mem.graduation_status(lid) == "quarantine"

    def test_idempotent_report(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        mem.add_lesson("p", "s", task_type="t")
        inj = mem.inject_for("p", "m", task_type="t", mode="shadow")
        cf_id = mem.enqueue_counterfactual(inj, "base")
        assert mem.report_counterfactual_outcome(cf_id, True, True) is True
        assert mem.report_counterfactual_outcome(cf_id, True, True) is False

    def test_unknown_cf_raises(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        with pytest.raises(KeyError):
            mem.report_counterfactual_outcome("nope", True, True)


class TestShadowRecoveryPath:
    def test_shadow_includes_quarantined_lesson(self, data_dir: Path) -> None:
        mem = AgentMemory(data_dir, trust=False)
        mem.add_lesson("extract dates", "use ISO-8601", task_type="extraction")
        # Live-quarantine it.
        for _ in range(8):
            inj = mem.inject_for("extract dates", "m", task_type="extraction")
            mem.report_outcome(inj, success=False)
        assert mem.stats()["lessons_quarantined"] == 1

        # Live path excludes it...
        live = mem.inject_for("extract dates", "m", task_type="extraction")
        assert "ISO-8601" not in live.text
        # ...but shadow path re-includes it for re-evaluation (recovery route).
        shadow = mem.inject_for("extract dates", "m", task_type="extraction", mode="shadow")
        assert "ISO-8601" in shadow.text
