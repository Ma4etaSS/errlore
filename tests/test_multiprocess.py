"""Cross-PROCESS concurrency tests.

The rest of the suite exercises concurrency with threads in a single process,
which share one in-process FileLock registry and one read cache. errlore's
durability claims are about *separate processes* (the flagship Claude Code
integration spawns a fresh process per hook), where the only coordination is
the on-disk file lock and each process has its own cache. These tests spawn
real OS processes against a shared data_dir to cover that path:

* lost updates across the read-modify-write cycle (atomic_update under the
  cross-process lock, reading fresh from disk rather than a stale cache);
* at-most-once report_outcome when many processes report the same handle.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

from errlore import AgentMemory

# Linux is the target; fork keeps the workers fast and avoids re-import cost.
_CTX = mp.get_context("fork")


def _log_resolve_worker(data_dir_str: str, proc_n: int, n_ops: int) -> None:
    mem = AgentMemory(Path(data_dir_str))
    for i in range(n_ops):
        eid = mem.log_error(f"model-{proc_n}", "load", f"Err{proc_n}x{i}: boom")
        # resolve() is a read-modify-write (atomic_update) -- the lost-update
        # path that a stale cross-process cache would corrupt.
        mem.resolve(eid, "fixed", lesson=None)


def _report_worker(data_dir_str: str, handle_id: str, q: mp.Queue[bool]) -> None:
    mem = AgentMemory(Path(data_dir_str))
    q.put(mem.report_outcome(handle_id, success=True))


class TestCrossProcessNoLostUpdates:
    N_PROC = 4
    N_OPS = 20

    def test_concurrent_processes_lose_no_errors(self, data_dir: Path) -> None:
        procs = [
            _CTX.Process(
                target=_log_resolve_worker, args=(str(data_dir), n, self.N_OPS)
            )
            for n in range(self.N_PROC)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, f"worker crashed: exitcode={p.exitcode}"

        expected = self.N_PROC * self.N_OPS
        stats = AgentMemory(data_dir).stats()
        assert stats["errors_total"] == expected, (
            f"lost updates across processes: "
            f"{expected - stats['errors_total']} of {expected} errors vanished"
        )
        assert stats["errors_resolved"] == expected


class TestCrossProcessReportIsAtMostOnce:
    N_PROC = 6

    def test_many_processes_report_same_handle_once(self, data_dir: Path) -> None:
        # Seed one lesson and issue a single injection for it.
        seed = AgentMemory(data_dir)
        eid = seed.log_error("gpt-x", "extraction", "BoomError: kaboom")
        seed.resolve(eid, "fixed", lesson="when kaboom, check the fuse first")
        inj = seed.inject_for("a kaboom task", model="gpt-x", task_type="extraction")
        assert inj.lesson_ids, "setup: lesson should be injected"
        lid = inj.lesson_ids[0]

        # Every process races to report the SAME handle.
        q: mp.Queue[bool] = _CTX.Queue()
        procs = [
            _CTX.Process(target=_report_worker, args=(str(data_dir), inj.handle_id, q))
            for _ in range(self.N_PROC)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, f"worker crashed: exitcode={p.exitcode}"

        returns = [q.get() for _ in range(self.N_PROC)]
        assert sum(1 for r in returns if r is True) == 1, (
            f"expected exactly one True report, got {returns}"
        )

        # And the lesson is reinforced exactly once, not N_PROC times.
        fresh = AgentMemory(data_dir)
        lesson = next(le for le in fresh.lessons() if le.id == lid)
        assert lesson.applied_count == 1, (
            f"lesson reinforced {lesson.applied_count}x across processes; "
            "at-most-once violated"
        )

        # Exactly one 'reported' marker on disk.
        import json

        markers = [
            json.loads(line)
            for line in (data_dir / "injections.jsonl").read_text().splitlines()
            if '"reported"' in line
        ]
        assert len(markers) == 1, f"expected 1 reported marker, got {len(markers)}"
