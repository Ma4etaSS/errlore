"""Durable JSONL I/O layer: writer, sidecar index, and auto-repair.

Public API:
    JSONLWriter  — cross-process append, batch append, atomic rewrite, rotation
    JSONLIndex   — sidecar .idx with byte offsets, tail-read, rebuild
    repair_file  — auto-repair corrupted JSONL (glued records, bad lines)
    repair_directory — batch repair all JSONL in a directory
    RepairStats  — statistics returned by repair operations
"""

from errlore.io.jsonl_index import JSONLIndex
from errlore.io.jsonl_writer import JSONLWriter
from errlore.io.repair import RepairStats, repair_directory, repair_file

__all__ = [
    "JSONLIndex",
    "JSONLWriter",
    "RepairStats",
    "repair_directory",
    "repair_file",
]
