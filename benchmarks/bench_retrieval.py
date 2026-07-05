#!/usr/bin/env python3
"""Benchmark: word-overlap vs embedding retrieval on a gold dataset.

Loads ``tests/gold/retrieval_gold.jsonl``, indexes all 40 lessons in a
:class:`~errlore.lessons.store.LessonStore`, and measures recall@k / MRR
for both retrieval strategies.

Exit code:
    0 -- embeddings recall@5 > word-overlap recall@5  (gate PASS)
    1 -- embeddings did NOT beat word-overlap          (gate FAIL)

Usage::

    .venv/bin/python benchmarks/bench_retrieval.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Gold data loader
# ---------------------------------------------------------------------------

_GOLD_PATH = Path(__file__).resolve().parent.parent / "tests" / "gold" / "retrieval_gold.jsonl"


def _load_gold() -> list[dict[str, str]]:
    """Read gold entries from JSONL."""
    entries: list[dict[str, str]] = []
    with open(_GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(
    gold: list[dict[str, str]],
    id_map: dict[int, str],
    search_fn: Any,  # Callable[[str], list[Lesson]]
    max_k: int = 5,
) -> dict[str, float]:
    """Compute recall@1, recall@3, recall@5, and MRR."""
    hits: dict[int, int] = {k: 0 for k in (1, 3, 5)}
    reciprocal_ranks: list[float] = []

    for idx, entry in enumerate(gold):
        target_id = id_map[idx]
        results = search_fn(entry["query"])
        result_ids = [le.id for le in results]

        # Reciprocal rank
        rr = 0.0
        for rank, rid in enumerate(result_ids[:max_k], 1):
            if rid == target_id:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        for k in hits:
            if target_id in result_ids[:k]:
                hits[k] += 1

    n = len(gold)
    return {
        "recall@1": hits[1] / n,
        "recall@3": hits[3] / n,
        "recall@5": hits[5] / n,
        "MRR": sum(reciprocal_ranks) / n,
    }


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _run_word_overlap(gold: list[dict[str, str]]) -> tuple[dict[str, float], dict[int, str]]:
    """Run benchmark with word-overlap retrieval (no retriever)."""
    from errlore.lessons.store import LessonStore

    with tempfile.TemporaryDirectory() as tmp:
        store = LessonStore(Path(tmp))
        id_map: dict[int, str] = {}

        for idx, entry in enumerate(gold):
            lid = store.log_lesson(
                pattern=entry["lesson_pattern"],
                solution=entry["lesson_solution"],
            )
            id_map[idx] = lid

        def search(query: str) -> list[Any]:
            return store.search_lessons(query=query, limit=5)

        metrics = _compute_metrics(gold, id_map, search)
    return metrics, id_map


def _run_embeddings(gold: list[dict[str, str]]) -> tuple[dict[str, float], dict[int, str]]:
    """Run benchmark with embedding retrieval (FastEmbedBackend + VectorIndex)."""
    from errlore.lessons.store import LessonStore
    from errlore.retrieval.backend import FastEmbedBackend
    from errlore.retrieval.index import VectorIndex

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        backend = FastEmbedBackend()
        index = VectorIndex(tmp_path, backend)
        store = LessonStore(tmp_path, retriever=index)
        id_map: dict[int, str] = {}

        for idx, entry in enumerate(gold):
            lid = store.log_lesson(
                pattern=entry["lesson_pattern"],
                solution=entry["lesson_solution"],
            )
            id_map[idx] = lid

        def search(query: str) -> list[Any]:
            return store.search_lessons(query=query, limit=5)

        metrics = _compute_metrics(gold, id_map, search)
    return metrics, id_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run both benchmarks, print results, return exit code."""
    gold = _load_gold()
    print(f"Gold dataset: {len(gold)} entries from {_GOLD_PATH.name}")
    print()

    print("Running word-overlap benchmark...")
    wo_metrics, _ = _run_word_overlap(gold)

    print("Running embeddings benchmark...")
    emb_metrics, _ = _run_embeddings(gold)

    # -- Pretty table -----------------------------------------------------
    header = f"{'metric':<16} {'word-overlap':>14} {'embeddings':>14}"
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for key in ("recall@1", "recall@3", "recall@5", "MRR"):
        wo_val = wo_metrics[key]
        emb_val = emb_metrics[key]
        print(f"{key:<16} {wo_val:>14.3f} {emb_val:>14.3f}")
    print(sep)
    print()

    # -- Gate check -------------------------------------------------------
    wo_r5 = wo_metrics["recall@5"]
    emb_r5 = emb_metrics["recall@5"]
    passed = emb_r5 > wo_r5

    status = "PASS" if passed else "FAIL"
    print(
        f"Gate: embeddings recall@5 ({emb_r5:.3f})"
        f" > word-overlap recall@5 ({wo_r5:.3f}): {status}",
    )

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
