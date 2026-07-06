"""Regression tests for bugs found during full end-to-end verification.

Bug 1: TrustEngine state silently reset on process restart (facade used the
       constructor, which never reads an existing state file).
Bug 2: Lost updates under concurrency -- resolve_error/reinforce/decay_unused
       did read -> atomic_rewrite outside a shared lock, so records appended
       by other threads between the read and the rewrite were wiped.
Bug 3: read_all cached a pre-append snapshot under the post-append stat when
       an append landed mid-parse, poisoning the cache; atomic_update then
       silently deleted the concurrent record.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from errlore import AgentMemory
from errlore.lessons.store import LessonStore


class TestTrustSurvivesRestart:
    def test_trust_state_restored_by_new_instance(self, data_dir: Path) -> None:
        mem1 = AgentMemory(data_dir)
        err_id = mem1.log_error("gpt-5.5", "extraction", "SomeError: boom")
        mem1.resolve(err_id, "fixed", lesson="when boom happens check the fuse")
        inj = mem1.inject_for("task with boom", model="gpt-5.5", task_type="extraction")
        mem1.report_outcome(inj, success=True)
        w1 = mem1.stats()["trust"]["gpt-5.5"]

        # Simulate a process restart: brand-new facade over the same dir.
        mem2 = AgentMemory(data_dir)
        stats2 = mem2.stats()
        assert "gpt-5.5" in stats2["trust"], "trust state lost on restart"
        assert abs(stats2["trust"]["gpt-5.5"] - w1) < 1e-9

        # And learning continues from where it left off.
        inj2 = mem2.inject_for(
            "another boom task", model="gpt-5.5", task_type="extraction"
        )
        assert inj2.lesson_ids, "lesson not found after restart"
        mem2.report_outcome(inj2, success=True)
        assert mem2.stats()["trust"]["gpt-5.5"] > w1


class TestNoLostUpdatesUnderConcurrency:
    N_THREADS = 8
    N_OPS = 25

    def test_concurrent_log_and_resolve_loses_nothing(self, data_dir: Path) -> None:
        """log_error appends race against resolve_error full-file rewrites."""
        mem = AgentMemory(data_dir)

        def worker(n: int) -> None:
            for i in range(self.N_OPS):
                eid = mem.log_error(f"model-{n}", "load", f"Err{n}x{i}: boom")
                mem.resolve(eid, "fixed", lesson=None)

        threads = [
            threading.Thread(target=worker, args=(n,)) for n in range(self.N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = self.N_THREADS * self.N_OPS
        stats = mem.stats()
        assert stats["errors_total"] == expected, (
            f"lost updates: {expected - stats['errors_total']} of {expected} "
            "errors disappeared"
        )
        assert stats["errors_resolved"] == expected

        # Every line on disk is valid JSON (no torn writes).
        for line in (data_dir / "errors.jsonl").read_text().splitlines():
            json.loads(line)

    def test_concurrent_log_lesson_and_reinforce(self, data_dir: Path) -> None:
        """log_lesson appends race against reinforce full-file rewrites."""
        store = LessonStore(data_dir)
        seed_id = store.log_lesson(
            pattern="seed pattern completely unlike the others",
            solution="seed solution",
        )
        n_new = 40

        def appender() -> None:
            for i in range(n_new):
                # Patterns made dissimilar enough to bypass fuzzy dedup.
                store.log_lesson(
                    pattern=f"unique{i} zebra{i * 7} quantum{i * 13} pattern",
                    solution=f"solution number {i}",
                )

        def reinforcer() -> None:
            for _ in range(n_new):
                store.reinforce(seed_id, success=True)

        t1 = threading.Thread(target=appender)
        t2 = threading.Thread(target=reinforcer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        counts = store.counts()
        assert counts["lessons_total"] == 1 + n_new, (
            f"lost lessons: expected {1 + n_new}, got {counts['lessons_total']}"
        )
        rows = [
            json.loads(line)
            for line in (data_dir / "lessons.jsonl").read_text().splitlines()
        ]
        seed_row = next(r for r in rows if r["id"] == seed_id)
        assert seed_row["applied_count"] == n_new


class TestReadCacheNotPoisonedByMidParseAppend:
    """Bug 3: read_all cached pre-append records under the post-append stat.

    An append landing between the parse loop and the re-stat made read_all
    store stale records keyed by the fresh mtime/size; every later read
    returned stale data and atomic_update silently deleted the concurrent
    record.  Fixed by caching only when pre-parse stat == post-parse stat.
    """

    def test_append_during_parse_does_not_poison_cache(
        self, data_dir: Path, monkeypatch: object
    ) -> None:
        import io as _io

        import errlore.io.jsonl_writer as jw

        writer = jw.JSONLWriter()
        path = data_dir / "records.jsonl"
        writer.append(path, {"id": 1})

        real_open = open
        raced = {"done": False}

        def racing_open(file: object, mode: str = "r", *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            # Intercept only read_all's read of our file, exactly once:
            # snapshot the current content, append a record behind the
            # parser's back, then hand the parser the stale snapshot.
            if str(file) == str(path) and mode == "r" and not raced["done"]:
                raced["done"] = True
                content = path.read_text(encoding="utf-8")
                with real_open(path, "a", encoding="utf-8") as af:
                    af.write(json.dumps({"id": 2}) + "\n")
                return _io.StringIO(content)
            return real_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(jw, "open", racing_open, raising=False)  # type: ignore[attr-defined]

        # The racy read itself legitimately sees the pre-append snapshot...
        first = writer.read_all(path)
        assert [r["id"] for r in first] == [1]

        # ...but it must NOT have been cached: the next read sees both.
        second = writer.read_all(path)
        assert [r["id"] for r in second] == [1, 2], (
            "read cache was poisoned by a mid-parse append"
        )

        # And atomic_update must not delete the concurrently appended record.
        result = writer.atomic_update(path, lambda entries: entries)
        assert result is not None
        assert [r["id"] for r in result] == [1, 2]
