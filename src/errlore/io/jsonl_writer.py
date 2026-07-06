"""Cross-process durable JSONL writer with filelock.

Guarantees append atomicity via:
- filelock (cross-process locking)
- binary mode "ab" (atomic writes below PIPE_BUF)
- explicit fsync by default
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger("errlore.io")

_LOCK_TIMEOUT: float = 10.0
_REPLACE_RETRIES: int = 5
_REPLACE_BACKOFF: float = 0.5

_DEFAULT_MAX_BYTES: int = 50_000_000  # 50 MB
_DEFAULT_MAX_ARCHIVES: int = 3


class JSONLWriter:
    """Cross-process JSONL writer with filelock, rotation, and atomic rewrite.

    Args:
        lock_timeout: Timeout in seconds for acquiring file locks.
        max_bytes: Maximum file size before rotation.  ``None`` disables
            rotation entirely -- useful for files with ID-based lookups
            (errors, lessons, injections) where all records must remain
            visible.  Growth is handled by future log compaction.
        max_archives: Maximum number of rotated archive files to keep.
    """

    def __init__(
        self,
        lock_timeout: float = _LOCK_TIMEOUT,
        max_bytes: int | None = _DEFAULT_MAX_BYTES,
        max_archives: int = _DEFAULT_MAX_ARCHIVES,
    ) -> None:
        self._lock_timeout = lock_timeout
        self._max_bytes = max_bytes
        self._max_archives = max_archives
        self._locks: dict[str, FileLock] = {}
        self._lock_guard = threading.Lock()
        # C1: mtime+size read cache -- avoids O(n) re-parse on hot paths.
        # Key: str(path), value: (mtime_ns, size, records).
        self._read_cache: dict[str, tuple[int, int, list[dict[str, object]]]] = {}
        self._cache_lock = threading.Lock()

    def _get_lock(self, path: Path) -> FileLock:
        """Get or create a FileLock for the given path (thread-safe)."""
        key = str(path)
        if key not in self._locks:
            with self._lock_guard:
                # Double-checked: another thread may have created it.
                if key not in self._locks:
                    self._locks[key] = FileLock(
                        key + ".lock", timeout=self._lock_timeout,
                    )
        return self._locks[key]

    def lock(self, path: Path) -> FileLock:
        """Return the FileLock for *path* (public access for callers that
        need to wrap broader read-modify-write cycles under the same lock).

        The returned ``FileLock`` is re-entrant within the same thread, so
        nested ``append`` / ``read_all`` calls inside a ``with writer.lock(p):``
        block will not deadlock.
        """
        return self._get_lock(path)

    @staticmethod
    def _rotate_versioned_files(path: Path, max_archives: int) -> None:
        """Shift archive chain for a versioned file.

        Works for both *.jsonl and *.idx: current becomes .1,
        previous .1 becomes .2, etc.
        """
        for i in range(max_archives, 0, -1):
            src = path.with_suffix(f"{path.suffix}.{i}")
            dst = path.with_suffix(f"{path.suffix}.{i + 1}")
            if i == max_archives and src.exists():
                src.unlink()
            elif src.exists():
                src.replace(dst)

        if path.exists():
            path.replace(path.with_suffix(f"{path.suffix}.1"))

    def _rotate_if_needed(self, path: Path) -> None:
        """Rotate the JSONL file when it exceeds max_bytes.

        When ``max_bytes`` is ``None``, rotation is disabled.
        """
        if self._max_bytes is None:
            return
        if not path.exists() or path.stat().st_size < self._max_bytes:
            return
        idx_path = path.with_suffix(".idx")
        self._rotate_versioned_files(idx_path, self._max_archives)
        self._rotate_versioned_files(path, self._max_archives)

    def append(
        self,
        path: Path,
        entry: dict[str, object],
        *,
        fsync: bool = True,
    ) -> int:
        """Append a single record to a JSONL file (cross-process safe).

        Args:
            path: Path to the JSONL file.
            entry: Dictionary record to write.
            fsync: Force fsync after write (default True).

        Returns:
            Byte offset of the written line's start.
        """
        lock = self._get_lock(path)
        line = (json.dumps(entry, ensure_ascii=False, default=str) + "\n").encode("utf-8")

        with lock:
            self._rotate_if_needed(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as f:
                offset = f.tell()
                f.write(line)
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())
        return offset

    def append_batch(
        self,
        path: Path,
        entries: Sequence[dict[str, object]],
        *,
        fsync: bool = True,
    ) -> int:
        """Append multiple records under a single lock acquisition.

        Args:
            path: Path to the JSONL file.
            entries: Sequence of dict records.
            fsync: Force fsync after the entire batch (default True).

        Returns:
            Number of records written.
        """
        if not entries:
            return 0

        lines = b"".join(
            (json.dumps(e, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            for e in entries
        )
        lock = self._get_lock(path)

        with lock:
            self._rotate_if_needed(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as f:
                f.write(lines)
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())
        return len(entries)

    def atomic_rewrite(
        self,
        path: Path,
        entries: Sequence[dict[str, object]],
    ) -> None:
        """Atomically rewrite a JSONL file (filelock + tmp + os.replace).

        Also rebuilds the sidecar .idx file atomically.

        Args:
            path: Path to the JSONL file.
            entries: Full list of records to write.
        """
        from errlore.io.jsonl_index import JSONLIndex

        lock = self._get_lock(path)
        tmp_jsonl: str | None = None
        tmp_idx: str | None = None
        idx_path = path.with_suffix(".idx")

        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            data_fd, tmp_jsonl = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            idx_fd, tmp_idx = tempfile.mkstemp(dir=path.parent, suffix=".idx.tmp")
            try:
                with os.fdopen(data_fd, "wb") as data_f, os.fdopen(idx_fd, "wb") as idx_f:
                    offset = 0
                    for entry in entries:
                        line = (
                            json.dumps(entry, ensure_ascii=False, default=str) + "\n"
                        ).encode("utf-8")
                        data_f.write(line)
                        idx_f.write(struct.pack(">Q", offset))
                        offset += len(line)
                    data_f.flush()
                    os.fsync(data_f.fileno())
                    idx_f.flush()
                    os.fsync(idx_f.fileno())

                # Replace data file with retries (Windows Defender can hold handles)
                for attempt in range(_REPLACE_RETRIES):
                    try:
                        if tmp_jsonl is not None:
                            os.replace(tmp_jsonl, str(path))
                        tmp_jsonl = None
                        break
                    except PermissionError:
                        if attempt < _REPLACE_RETRIES - 1:
                            time.sleep(_REPLACE_BACKOFF * (attempt + 1))
                        else:
                            raise

                # Replace index file
                try:
                    if tmp_idx is not None:
                        os.replace(tmp_idx, str(idx_path))
                    tmp_idx = None
                except PermissionError:
                    # If sidecar locked, rebuild index from fresh JSONL
                    JSONLIndex(path).rebuild()
                    if tmp_idx is not None:
                        with contextlib.suppress(OSError):
                            os.unlink(tmp_idx)
                    tmp_idx = None
            finally:
                if tmp_jsonl is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_jsonl)
                if tmp_idx is not None:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_idx)

    def atomic_update(
        self,
        path: Path,
        transform: Callable[[list[dict[str, object]]], list[dict[str, object]] | None],
    ) -> list[dict[str, object]] | None:
        """Read-modify-write a JSONL file under ONE file lock (no lost updates).

        ``atomic_rewrite`` alone is not race-safe for read-modify-write cycles:
        if a caller reads the file, computes new content, and then rewrites,
        any record appended by another thread/process in between is silently
        lost.  ``atomic_update`` holds the file lock across the entire
        read -> transform -> replace sequence, so concurrent ``append`` calls
        (which take the same lock) serialize correctly.

        Args:
            path: Path to the JSONL file.
            transform: Callback receiving the current records; returns the new
                full record list, or None to abort without writing.

        Returns:
            The new record list that was written, or None if aborted.
        """
        lock = self._get_lock(path)
        with lock:  # FileLock is re-entrant within the same thread
            entries = self.read_all(path)
            new_entries = transform(entries)
            if new_entries is None:
                return None
            self.atomic_rewrite(path, new_entries)
            return new_entries

    def read_all(self, path: Path) -> list[dict[str, object]]:
        """Read all valid records from a JSONL file.

        Results are cached by (mtime_ns, size).  Subsequent calls that hit the
        cache skip JSON parsing entirely.  Callers receive shallow copies of
        the cached records so mutations do not corrupt the cache.

        Args:
            path: Path to the JSONL file.

        Returns:
            List of parsed dict records.
        """
        if not path.exists():
            return []

        st = path.stat()
        key = str(path)

        with self._cache_lock:
            cached = self._read_cache.get(key)
            if cached is not None:
                c_mtime, c_size, c_records = cached
                if c_mtime == st.st_mtime_ns and c_size == st.st_size:
                    return [dict(r) for r in c_records]

        # Cache miss -- parse from disk.
        records: list[dict[str, object]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        records.append(obj)
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupted line in %s", path.name)

        # Cache only if the file is unchanged since the pre-parse stat.
        # An append landing mid-parse would otherwise store the pre-append
        # records under the post-append mtime/size — a poisoned cache that
        # makes atomic_update silently drop the concurrent record.
        try:
            st2 = path.stat()
        except OSError:
            return records

        if st2.st_mtime_ns == st.st_mtime_ns and st2.st_size == st.st_size:
            with self._cache_lock:
                self._read_cache[key] = (st.st_mtime_ns, st.st_size, records)

        return [dict(r) for r in records]
