"""JSONL sidecar index for O(1) access to records by byte offset.

Each JSONL file can have a companion .idx file storing 8-byte big-endian
unsigned offsets. This enables:
- get_recent() via tail-read without scanning the entire file
- O(1) append (add offset to end of .idx)
- rebuild when the index is corrupted
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import struct
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger("errlore.io")

# Each entry in .idx = 8 bytes (unsigned long long, big-endian)
_OFFSET_FORMAT = ">Q"
_OFFSET_SIZE = struct.calcsize(_OFFSET_FORMAT)


class JSONLIndex:
    """Byte-offset index for a JSONL file."""

    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.idx_path = self.jsonl_path.with_suffix(".idx")
        self._lock = threading.Lock()

    def append_offset(self, byte_offset: int) -> None:
        """Append an offset for a new record to the index."""
        with self._lock:
            self.idx_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.idx_path, "ab") as f:
                f.write(struct.pack(_OFFSET_FORMAT, byte_offset))

    def _read_all_offsets(self) -> list[int]:
        """Read all valid offsets from .idx, discarding any truncated tail."""
        if not self.idx_path.exists():
            return []

        file_size = self.idx_path.stat().st_size
        if file_size == 0:
            return []

        aligned_size = file_size - (file_size % _OFFSET_SIZE)
        if aligned_size != file_size:
            logger.warning(
                "Index file %s has truncated tail (%d bytes ignored)",
                self.idx_path.name,
                file_size - aligned_size,
            )

        if aligned_size == 0:
            return []

        offsets: list[int] = []
        with open(self.idx_path, "rb") as f:
            data = f.read(aligned_size)
        for i in range(0, aligned_size, _OFFSET_SIZE):
            offsets.append(struct.unpack(_OFFSET_FORMAT, data[i : i + _OFFSET_SIZE])[0])
        return offsets

    def get_recent_offsets(self, count: int) -> list[int]:
        """Get byte-offsets of the last N records."""
        if count <= 0:
            return []
        with self._lock:
            offsets = self._read_all_offsets()
            total_entries = len(offsets)
            if total_entries == 0:
                return []
            start = max(0, total_entries - count)
            return offsets[start:]

    def read_records_at_offsets(self, offsets: list[int]) -> list[dict[str, object]]:
        """Read records from the JSONL file at specified byte-offsets."""
        if not self.jsonl_path.exists():
            return []
        records: list[dict[str, object]] = []
        with open(self.jsonl_path, encoding="utf-8") as f:
            for offset in offsets:
                try:
                    f.seek(offset)
                    line = f.readline()
                    if line.strip():
                        records.append(json.loads(line))
                except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning("Corrupted record at offset %d: %s", offset, e)
        return records

    def get_recent_records(self, count: int) -> list[dict[str, object]]:
        """Get the last N records via the index. O(1) per seek."""
        offsets = self.get_recent_offsets(count)
        if not offsets:
            return []
        return self.read_records_at_offsets(offsets)

    def entry_count(self) -> int:
        """Number of entries in the index."""
        return len(self._read_all_offsets())

    def rebuild(self) -> int:
        """Rebuild the index from the JSONL file.

        Returns:
            Number of indexed entries.
        """
        if not self.jsonl_path.exists():
            if self.idx_path.exists():
                try:
                    self.idx_path.unlink()
                except OSError:
                    logger.warning("Failed to remove stale index file %s", self.idx_path)
            return 0

        count = 0
        tmp_fd, tmp_path_str = tempfile.mkstemp(dir=self.idx_path.parent, suffix=".idx.tmp")
        try:
            with self._lock, os.fdopen(tmp_fd, "wb") as idx_f:
                with open(self.jsonl_path, encoding="utf-8") as jsonl_f:
                    while True:
                        offset = jsonl_f.tell()
                        line = jsonl_f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            json.loads(line)
                            idx_f.write(struct.pack(_OFFSET_FORMAT, offset))
                            count += 1
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping corrupted line at offset %d during rebuild",
                                offset,
                            )
                idx_f.flush()
                os.fsync(idx_f.fileno())
            os.replace(tmp_path_str, str(self.idx_path))
            tmp_path_str = ""
        finally:
            if tmp_path_str:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path_str)
        logger.info("Rebuilt index for %s: %d entries", self.jsonl_path.name, count)
        return count

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify index integrity against the JSONL file.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: list[str] = []

        if not self.idx_path.exists():
            if self.jsonl_path.exists() and self.jsonl_path.stat().st_size > 0:
                errors.append("Index file missing but JSONL has data")
            return len(errors) == 0, errors

        idx_size = self.idx_path.stat().st_size
        if idx_size % _OFFSET_SIZE != 0:
            errors.append(f"Index file size {idx_size} not aligned to {_OFFSET_SIZE} bytes")

        if not self.jsonl_path.exists():
            errors.append("JSONL file missing but index exists")
            return False, errors

        jsonl_size = self.jsonl_path.stat().st_size
        offsets = self.get_recent_offsets(self.entry_count())
        previous_offset = -1
        for offset in offsets:
            if offset < previous_offset:
                errors.append("Offsets are not monotonically increasing")
            previous_offset = offset
            if offset >= jsonl_size:
                errors.append(f"Offset {offset} exceeds JSONL size {jsonl_size}")
                continue

            try:
                with open(self.jsonl_path, encoding="utf-8") as f:
                    f.seek(offset)
                    line = f.readline()
                if not line.strip():
                    errors.append(f"Offset {offset} points to empty record")
                    continue
                json.loads(line)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                errors.append(f"Offset {offset} is invalid: {exc}")

        return len(errors) == 0, errors
