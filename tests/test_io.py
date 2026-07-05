"""Tests for errlore.io — JSONL writer, index, and repair."""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from errlore.io import JSONLIndex, JSONLWriter, repair_directory, repair_file

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def writer() -> JSONLWriter:
    """Fresh JSONLWriter with small rotation threshold for testing."""
    return JSONLWriter(max_bytes=500, max_archives=2)


@pytest.fixture()
def jsonl_path(data_dir: Path) -> Path:
    return data_dir / "test.jsonl"


# ---------------------------------------------------------------------------
# append + read round-trip
# ---------------------------------------------------------------------------


class TestAppend:
    def test_append_and_read_roundtrip(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        """Appended record can be read back identically."""
        entry = {"level": "error", "msg": "disk full", "code": 42}
        offset = writer.append(jsonl_path, entry)
        assert offset == 0

        records = writer.read_all(jsonl_path)
        assert len(records) == 1
        assert records[0] == entry

    def test_append_returns_correct_offset(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        """Second append starts at the byte after first line."""
        e1 = {"a": 1}
        e2 = {"b": 2}
        off1 = writer.append(jsonl_path, e1)
        off2 = writer.append(jsonl_path, e2)
        assert off1 == 0
        line1 = (json.dumps(e1, ensure_ascii=False) + "\n").encode("utf-8")
        assert off2 == len(line1)

    def test_append_fsync_false(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        """fsync=False still writes data (just without forced flush)."""
        entry = {"fast": True}
        writer.append(jsonl_path, entry, fsync=False)
        records = writer.read_all(jsonl_path)
        assert records == [entry]

    def test_append_creates_parent_dirs(self, data_dir: Path) -> None:
        """Writer creates missing parent directories."""
        w = JSONLWriter()
        deep_path = data_dir / "sub" / "deep" / "log.jsonl"
        w.append(deep_path, {"nested": True})
        assert deep_path.exists()
        assert w.read_all(deep_path) == [{"nested": True}]


# ---------------------------------------------------------------------------
# append_batch
# ---------------------------------------------------------------------------


class TestAppendBatch:
    def test_batch_writes_all(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        entries = [{"i": i} for i in range(10)]
        count = writer.append_batch(jsonl_path, entries)
        assert count == 10
        records = writer.read_all(jsonl_path)
        assert records == entries

    def test_batch_empty_list(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        count = writer.append_batch(jsonl_path, [])
        assert count == 0
        assert not jsonl_path.exists()

    def test_batch_fsync_param(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        entries = [{"x": 1}, {"x": 2}]
        writer.append_batch(jsonl_path, entries, fsync=False)
        assert writer.read_all(jsonl_path) == entries


# ---------------------------------------------------------------------------
# atomic_rewrite
# ---------------------------------------------------------------------------


class TestAtomicRewrite:
    def test_rewrite_replaces_content(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        """After atomic_rewrite, only the new entries remain."""
        writer.append_batch(jsonl_path, [{"old": i} for i in range(5)])
        new_entries: list[dict[str, object]] = [{"new": i} for i in range(3)]
        writer.atomic_rewrite(jsonl_path, new_entries)
        assert writer.read_all(jsonl_path) == new_entries

    def test_rewrite_creates_idx(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        """atomic_rewrite also writes a correct .idx sidecar."""
        entries: list[dict[str, object]] = [{"k": i} for i in range(4)]
        writer.atomic_rewrite(jsonl_path, entries)
        idx = JSONLIndex(jsonl_path)
        assert idx.entry_count() == 4
        assert idx.get_recent_records(4) == entries

    def test_rewrite_failure_preserves_original(
        self, writer: JSONLWriter, jsonl_path: Path
    ) -> None:
        """If an exception is raised during serialization, the old file stays intact."""
        original: list[dict[str, object]] = [{"orig": 1}]
        writer.append_batch(jsonl_path, original)

        class BombError(Exception):
            pass

        class Bomb:
            """Object that raises on JSON serialization."""
            def __repr__(self) -> str:
                raise BombError("boom")

        # The writer uses default=str, so we need to break at a different level.
        # We'll monkeypatch json.dumps to fail mid-stream.
        import errlore.io.jsonl_writer as writer_mod

        call_count = 0
        original_dumps = json.dumps

        def failing_dumps(*args: object, **kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise BombError("simulated crash")
            return original_dumps(*args, **kwargs)  # type: ignore[arg-type]

        writer_mod.json.dumps = failing_dumps  # type: ignore[assignment]
        try:
            with pytest.raises(BombError):
                writer.atomic_rewrite(jsonl_path, [{"a": 1}, {"b": 2}])
        finally:
            writer_mod.json.dumps = original_dumps  # type: ignore[assignment]

        # Original file must still be intact
        assert writer.read_all(jsonl_path) == original


# ---------------------------------------------------------------------------
# Index: offsets + tail-read
# ---------------------------------------------------------------------------


class TestIndex:
    def test_offsets_are_correct(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        """Index offsets match actual byte positions in the JSONL file."""
        entries: list[dict[str, object]] = [{"i": i, "data": "x" * i} for i in range(5)]
        idx = JSONLIndex(jsonl_path)

        for entry in entries:
            offset = writer.append(jsonl_path, entry)
            idx.append_offset(offset)

        all_offsets = idx.get_recent_offsets(100)
        assert len(all_offsets) == 5

        # Verify each offset points to the correct record
        with open(jsonl_path, encoding="utf-8") as f:
            for i, off in enumerate(all_offsets):
                f.seek(off)
                line = f.readline()
                record = json.loads(line)
                assert record == entries[i]

    def test_tail_read_matches_full_read(
        self, writer: JSONLWriter, jsonl_path: Path
    ) -> None:
        """get_recent_records returns the same as full-file read for last N."""
        entries: list[dict[str, object]] = [{"seq": i} for i in range(20)]
        idx = JSONLIndex(jsonl_path)

        for entry in entries:
            offset = writer.append(jsonl_path, entry)
            idx.append_offset(offset)

        last_5_via_index = idx.get_recent_records(5)
        all_records = writer.read_all(jsonl_path)
        last_5_via_full = all_records[-5:]

        assert last_5_via_index == last_5_via_full

    def test_rebuild_index(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        """rebuild() creates a valid index from the JSONL file alone."""
        entries: list[dict[str, object]] = [{"r": i} for i in range(7)]
        writer.append_batch(jsonl_path, entries)

        idx = JSONLIndex(jsonl_path)
        # No index yet
        assert idx.entry_count() == 0

        count = idx.rebuild()
        assert count == 7
        assert idx.get_recent_records(7) == entries

    def test_verify_integrity_valid(self, writer: JSONLWriter, jsonl_path: Path) -> None:
        """A freshly rebuilt index passes integrity check."""
        entries: list[dict[str, object]] = [{"v": i} for i in range(3)]
        writer.append_batch(jsonl_path, entries)
        idx = JSONLIndex(jsonl_path)
        idx.rebuild()
        is_valid, errors = idx.verify_integrity()
        assert is_valid
        assert errors == []

    def test_truncated_idx_handled(self, data_dir: Path) -> None:
        """Index with truncated tail bytes still reads valid entries."""
        jsonl_path = data_dir / "trunc.jsonl"
        idx_path = jsonl_path.with_suffix(".idx")

        # Write 3 full offsets + 4 garbage bytes
        with open(idx_path, "wb") as f:
            for i in range(3):
                f.write(struct.pack(">Q", i * 10))
            f.write(b"\x00\x01\x02\x03")

        idx = JSONLIndex(jsonl_path)
        offsets = idx.get_recent_offsets(100)
        assert len(offsets) == 3


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


class TestRepair:
    def test_repair_glued_records(self, data_dir: Path) -> None:
        """Glued JSON records (missing newline) are separated."""
        path = data_dir / "glued.jsonl"
        # Write two JSON objects glued together on one line
        line = json.dumps({"a": 1}) + json.dumps({"b": 2}) + "\n"
        path.write_text(line + json.dumps({"c": 3}) + "\n", encoding="utf-8")

        stats = repair_file(path)
        assert stats.fixed >= 2
        assert stats.dropped == 0

        # All three records survive
        w = JSONLWriter()
        records = w.read_all(path)
        values = [next(iter(r.keys())) for r in records]
        assert "a" in values
        assert "b" in values
        assert "c" in values

    def test_repair_drops_garbage(self, data_dir: Path) -> None:
        """Completely invalid lines are dropped."""
        path = data_dir / "garbage.jsonl"
        content = (
            json.dumps({"ok": 1}) + "\n"
            + "this is not json at all\n"
            + json.dumps({"ok": 2}) + "\n"
            + "\xff\xfe broken bytes will be written separately\n"
        )
        path.write_bytes(content.encode("utf-8", errors="surrogatepass"))

        # Inject actual invalid UTF-8 bytes
        with open(path, "ab") as f:
            f.write(b"\xff\xfe not utf8\n")

        stats = repair_file(path)
        assert stats.dropped >= 2
        assert stats.ok == 2

        w = JSONLWriter()
        records = w.read_all(path)
        assert len(records) == 2
        assert records[0] == {"ok": 1}
        assert records[1] == {"ok": 2}

    def test_repair_valid_file_unchanged(self, data_dir: Path) -> None:
        """A valid file is not rewritten (no fixed, no dropped)."""
        path = data_dir / "clean.jsonl"
        entries = [{"clean": i} for i in range(5)]
        path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8",
        )
        mtime_before = path.stat().st_mtime_ns

        stats = repair_file(path)
        assert stats.fixed == 0
        assert stats.dropped == 0
        assert stats.ok == 5

        # File should not have been rewritten
        mtime_after = path.stat().st_mtime_ns
        assert mtime_before == mtime_after

    def test_repair_dry_run(self, data_dir: Path) -> None:
        """dry_run=True does not modify the file."""
        path = data_dir / "dryrun.jsonl"
        glued = json.dumps({"x": 1}) + json.dumps({"y": 2}) + "\n"
        path.write_text(glued, encoding="utf-8")
        original_content = path.read_bytes()

        stats = repair_file(path, dry_run=True)
        assert stats.fixed >= 2
        assert path.read_bytes() == original_content

    def test_repair_rebuilds_index(self, data_dir: Path) -> None:
        """After repair, the sidecar .idx is consistent."""
        path = data_dir / "indexed.jsonl"
        entries = [json.dumps({"n": i}) for i in range(4)]
        # Inject one bad line
        lines = [*entries[:2], "not json", *entries[2:]]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        stats = repair_file(path)
        assert stats.dropped == 1
        assert stats.index_rebuilt == 4  # 4 valid records indexed

        idx = JSONLIndex(path)
        is_valid, errors = idx.verify_integrity()
        assert is_valid, errors


# ---------------------------------------------------------------------------
# Rotation by size
# ---------------------------------------------------------------------------


class TestRotation:
    def test_rotation_triggers_on_size(self, data_dir: Path) -> None:
        """File is rotated when it exceeds max_bytes."""
        path = data_dir / "rotate.jsonl"
        # max_bytes=500 in the writer fixture is too coupled, use explicit
        w = JSONLWriter(max_bytes=200, max_archives=2)

        # Write enough to exceed 200 bytes
        for i in range(20):
            w.append(path, {"i": i, "payload": "x" * 50})

        # After rotation, current file should be small
        assert path.exists()
        assert path.stat().st_size < 200

        # Archive .1 should exist
        archive1 = path.with_suffix(".jsonl.1")
        assert archive1.exists()

    def test_rotation_max_archives_respected(self, data_dir: Path) -> None:
        """Only max_archives rotated files are kept."""
        path = data_dir / "rot.jsonl"
        w = JSONLWriter(max_bytes=100, max_archives=2)

        # Write a lot to trigger multiple rotations
        for i in range(50):
            w.append(path, {"i": i, "data": "y" * 80})

        # .1 and .2 should exist, .3 should not
        assert path.with_suffix(".jsonl.1").exists()
        assert path.with_suffix(".jsonl.2").exists()
        assert not path.with_suffix(".jsonl.3").exists()


# ---------------------------------------------------------------------------
# Concurrent writes
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_writes_no_corruption(self, data_dir: Path) -> None:
        """Two threads writing concurrently produce no corrupted lines."""
        path = data_dir / "concurrent.jsonl"
        w = JSONLWriter(max_bytes=50_000_000)  # No rotation during test
        n_per_thread = 50
        errors: list[str] = []

        def writer_thread(thread_id: int) -> None:
            try:
                for i in range(n_per_thread):
                    w.append(path, {"thread": thread_id, "seq": i})
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [
            threading.Thread(target=writer_thread, args=(tid,))
            for tid in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

        # Verify all lines are valid JSON
        records = w.read_all(path)
        assert len(records) == 4 * n_per_thread

        # Verify all entries are present
        seen = set()
        for r in records:
            seen.add((r["thread"], r["seq"]))
        assert len(seen) == 4 * n_per_thread


# ---------------------------------------------------------------------------
# A1: Rotation disabled (max_bytes=None)
# ---------------------------------------------------------------------------


class TestRotationDisabled:
    def test_no_rotation_when_max_bytes_none(self, data_dir: Path) -> None:
        """Appending past the default limit with max_bytes=None never rotates."""
        path = data_dir / "no_rotate.jsonl"
        w = JSONLWriter(max_bytes=None)

        # Write enough to normally trigger rotation at 200 bytes.
        for i in range(40):
            w.append(path, {"i": i, "payload": "x" * 50})

        assert path.exists()
        # No archives should be created.
        assert not path.with_suffix(".jsonl.1").exists()

        # All records are still accessible.
        records = w.read_all(path)
        assert len(records) == 40

    def test_resolve_by_id_after_large_write(self, data_dir: Path) -> None:
        """Records addressable by ID remain visible after writing past the
        byte limit when rotation is disabled."""
        path = data_dir / "ids.jsonl"
        w = JSONLWriter(max_bytes=None)

        # Write an early record, then write a lot more.
        early = {"id": "early_record", "data": "important"}
        w.append(path, early)
        for i in range(50):
            w.append(path, {"id": f"filler_{i}", "data": "x" * 100})

        records = w.read_all(path)
        found = [r for r in records if r.get("id") == "early_record"]
        assert len(found) == 1
        assert found[0] == early


# ---------------------------------------------------------------------------
# C1: read_all mtime+size cache
# ---------------------------------------------------------------------------


class TestReadCache:
    def test_second_read_uses_cache(
        self, data_dir: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Second read_all hits cache -- no JSON parsing."""
        path = data_dir / "cached.jsonl"
        w = JSONLWriter(max_bytes=None)
        w.append(path, {"x": 1})
        w.append(path, {"x": 2})

        # First read: populates cache.
        r1 = w.read_all(path)
        assert len(r1) == 2

        # Patch json.loads to count calls.
        import errlore.io.jsonl_writer as wmod
        real_loads = json.loads
        counter = MagicMock(side_effect=real_loads)
        monkeypatch.setattr(wmod.json, "loads", counter)

        # Second read: should hit cache, zero json.loads calls.
        r2 = w.read_all(path)
        assert len(r2) == 2
        assert counter.call_count == 0

    def test_cache_invalidated_after_append(self, data_dir: Path) -> None:
        """After an append, the next read_all re-parses from disk."""
        path = data_dir / "inval.jsonl"
        w = JSONLWriter(max_bytes=None)
        w.append(path, {"v": 1})
        r1 = w.read_all(path)
        assert len(r1) == 1

        w.append(path, {"v": 2})
        r2 = w.read_all(path)
        assert len(r2) == 2

    def test_cache_returns_copies(self, data_dir: Path) -> None:
        """Mutating a returned record does not corrupt the cache."""
        path = data_dir / "copy.jsonl"
        w = JSONLWriter(max_bytes=None)
        w.append(path, {"k": "original"})

        r1 = w.read_all(path)
        r1[0]["k"] = "mutated"

        r2 = w.read_all(path)
        assert r2[0]["k"] == "original"


# ---------------------------------------------------------------------------
# C5: Additional repair tests
# ---------------------------------------------------------------------------


class TestRepairExtended:
    def test_repair_glued_and_garbage_mixed(self, data_dir: Path) -> None:
        """Glued `}{` plus trailing garbage -> fixed/dropped correct."""
        path = data_dir / "mixed.jsonl"
        glued = json.dumps({"a": 1}) + json.dumps({"b": 2})
        garbage = "not json at all"
        path.write_text(glued + "\n" + garbage + "\n", encoding="utf-8")

        stats = repair_file(path)
        assert stats.fixed >= 2  # glued records separated
        assert stats.dropped >= 1  # garbage line dropped

        w = JSONLWriter()
        records = w.read_all(path)
        keys = {next(iter(r.keys())) for r in records}
        assert "a" in keys
        assert "b" in keys

    def test_repair_directory_recursive(self, data_dir: Path) -> None:
        """repair_directory with recursive=True finds nested files."""
        sub = data_dir / "sub"
        sub.mkdir()
        p1 = data_dir / "top.jsonl"
        p2 = sub / "nested.jsonl"

        p1.write_text(json.dumps({"ok": 1}) + "\n", encoding="utf-8")
        glued = json.dumps({"x": 1}) + json.dumps({"y": 2}) + "\n"
        p2.write_text(glued, encoding="utf-8")

        results = repair_directory(data_dir, recursive=True)
        assert len(results) >= 2
        nested_key = str(p2)
        assert nested_key in results
        assert results[nested_key].fixed >= 2

    def test_repair_dry_run_no_change(self, data_dir: Path) -> None:
        """dry_run=True does not modify the file."""
        path = data_dir / "dry.jsonl"
        glued = json.dumps({"a": 1}) + json.dumps({"b": 2}) + "\n"
        path.write_text(glued, encoding="utf-8")
        original = path.read_bytes()

        stats = repair_file(path, dry_run=True)
        assert stats.fixed >= 2
        assert path.read_bytes() == original

    def test_verify_integrity_corrupt_idx_then_rebuild(self, data_dir: Path) -> None:
        """Corrupted .idx fails verify, rebuild fixes it."""
        path = data_dir / "bad_idx.jsonl"
        w = JSONLWriter()
        entries: list[dict[str, object]] = [{"v": i} for i in range(3)]
        w.append_batch(path, entries)

        idx = JSONLIndex(path)
        # Write garbage to .idx
        idx.idx_path.write_bytes(b"\xff" * 40)

        is_valid, errors = idx.verify_integrity()
        assert not is_valid
        assert len(errors) > 0

        # Rebuild should fix it.
        idx.rebuild()
        is_valid2, errors2 = idx.verify_integrity()
        assert is_valid2, f"Errors after rebuild: {errors2}"
