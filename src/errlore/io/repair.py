"""Auto-repair for corrupted JSONL files.

Scans JSONL files, recovers glued JSON records (missing newlines),
drops unrecoverable lines, and rebuilds the sidecar index.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from pathlib import Path

from errlore.io.jsonl_index import JSONLIndex
from errlore.io.jsonl_writer import JSONLWriter

logger = logging.getLogger("errlore.io")

# Pattern for glued JSON objects: `}{` without a newline between them
_GLUED_PATTERN = re.compile(r"\}\s*\{")


def _try_parse(line: str) -> list[dict[str, object]]:
    """Attempt to parse a line as JSON. Handles glued records."""
    line = line.strip()
    if not line:
        return []

    # Attempt 1: standard JSON
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return [obj]
        return []
    except json.JSONDecodeError:
        pass

    # Attempt 2: glued JSON (`}{` -> `}\n{`)
    if _GLUED_PATTERN.search(line):
        parts = _GLUED_PATTERN.sub("}\n{", line).split("\n")
        results: list[dict[str, object]] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                obj = json.loads(part)
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                pass
        if results:
            return results

    return []


class RepairStats:
    """Statistics from a JSONL repair operation."""

    __slots__ = ("dropped", "fixed", "index_rebuilt", "ok", "total_lines")

    def __init__(self) -> None:
        self.ok: int = 0
        self.fixed: int = 0
        self.dropped: int = 0
        self.total_lines: int = 0
        self.index_rebuilt: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return stats as a plain dict."""
        return {
            "ok": self.ok,
            "fixed": self.fixed,
            "dropped": self.dropped,
            "total_lines": self.total_lines,
            "index_rebuilt": self.index_rebuilt,
        }


def repair_file(
    path: Path,
    *,
    dry_run: bool = False,
    writer: JSONLWriter | None = None,
) -> RepairStats:
    """Repair a single JSONL file.

    Recovers glued records, drops unrecoverable lines, then atomically
    rewrites the file and rebuilds the sidecar index.

    Args:
        path: Path to the JSONL file.
        dry_run: If True, only report what would happen.
        writer: Optional JSONLWriter instance (created internally if None).

    Returns:
        RepairStats with counts of ok/fixed/dropped lines.
    """
    stats = RepairStats()
    repaired: list[dict[str, object]] = []

    if not path.exists():
        # Clean up orphan index if JSONL is gone
        idx = JSONLIndex(path)
        if not dry_run and idx.idx_path.exists():
            with contextlib.suppress(OSError):
                idx.idx_path.unlink()
        return stats

    w = writer or JSONLWriter()

    # A3: hold the file lock for the entire read -> parse -> rewrite cycle
    # so that concurrent appends are not lost between read and rewrite.
    with w.lock(path):
        try:
            raw = path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return stats

        for line_bytes in raw.split(b"\n"):
            stats.total_lines += 1
            try:
                line = line_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                stats.dropped += 1
                continue

            if not line:
                continue

            parsed = _try_parse(line)
            if not parsed:
                stats.dropped += 1
                continue

            if len(parsed) == 1:
                # Single JSON per line
                try:
                    json.loads(line)
                    stats.ok += 1
                except json.JSONDecodeError:
                    stats.fixed += 1
            else:
                # Glued records separated
                stats.fixed += len(parsed)

            repaired.extend(parsed)

        if not dry_run and (stats.fixed > 0 or stats.dropped > 0):
            w.atomic_rewrite(path, repaired)

        if not dry_run:
            idx = JSONLIndex(path)
            stats.index_rebuilt = idx.rebuild()

    return stats


def repair_directory(
    directory: Path,
    *,
    dry_run: bool = False,
    recursive: bool = True,
) -> dict[str, RepairStats]:
    """Repair all JSONL files in a directory.

    Args:
        directory: Directory to scan.
        dry_run: If True, only report what would happen.
        recursive: If True, scan subdirectories too.

    Returns:
        Mapping of filename to RepairStats.
    """
    if not directory.is_dir():
        logger.error("Not a directory: %s", directory)
        return {}

    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    files = sorted(directory.glob(pattern))

    # Deduplicate resolved paths
    seen: set[str] = set()
    unique: list[Path] = []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)

    writer = JSONLWriter()
    results: dict[str, RepairStats] = {}
    for path in unique:
        results[str(path)] = repair_file(path, dry_run=dry_run, writer=writer)

    return results
