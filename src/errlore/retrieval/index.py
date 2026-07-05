"""Brute-force cosine similarity vector index backed by numpy.

Persistence: ``data_dir/vectors.npy`` + ``data_dir/vector_meta.json``.
When the stored ``model_id`` or ``dim`` no longer matches the active
backend, the on-disk index is discarded and rebuilt incrementally via
``add`` / ``add_batch`` calls from :class:`~errlore.lessons.store.LessonStore`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from errlore.retrieval import EmbeddingBackend

logger = logging.getLogger("errlore.retrieval")


class VectorIndex:
    """Thread-safe vector index with incremental persistence.

    Implements the :class:`~errlore.retrieval.LessonRetriever` protocol
    (``search``, ``add``, ``remove``).

    Args:
        data_dir: Directory for ``vectors.npy`` and ``vector_meta.json``.
        backend: Embedding provider satisfying ``EmbeddingBackend``.
    """

    def __init__(self, data_dir: Path | str, backend: EmbeddingBackend) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend
        self._vectors_path = self._data_dir / "vectors.npy"
        self._meta_path = self._data_dir / "vector_meta.json"

        self._ids: list[str] = []
        self._id_set: set[str] = set()
        self._vectors: npt.NDArray[np.floating] | None = None  # shape (N, dim), L2-normalized
        self._lock = threading.Lock()

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load persisted vectors and metadata from disk.

        Corrupt or unreadable files are treated as an empty index with a
        warning -- the index will be rebuilt incrementally on the next
        ``add`` / ``add_batch`` call from :class:`LessonStore`.
        """
        if not self._meta_path.exists() or not self._vectors_path.exists():
            return

        try:
            with open(self._meta_path, encoding="utf-8") as f:
                meta: dict[str, object] = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Cannot read vector metadata %s: %s", self._meta_path, exc)
            return

        stored_model = meta.get("model_id")
        stored_dim = meta.get("dim")

        # B5-ext: fastembed version mismatch triggers rebuild.
        stored_fe_version = meta.get("fastembed_version")
        active_fe_version = self._get_fastembed_version()
        if stored_fe_version is not None and stored_fe_version != active_fe_version:
            logger.warning(
                "fastembed version mismatch (stored=%s, active=%s); "
                "index will be rebuilt on next sync",
                stored_fe_version,
                active_fe_version,
            )
            return

        if stored_model != self._backend.model_id or stored_dim != self._backend.dim:
            logger.warning(
                "Index model/dim mismatch (stored=%s/%s, active=%s/%s); "
                "index will be rebuilt on next sync",
                stored_model,
                stored_dim,
                self._backend.model_id,
                self._backend.dim,
            )
            return

        raw_ids = meta.get("ids")
        ids: list[str] = [str(x) for x in raw_ids] if isinstance(raw_ids, list) else []

        try:
            arr: npt.NDArray[np.floating] = np.load(self._vectors_path)
        except (OSError, ValueError, EOFError) as exc:
            logger.warning("Cannot read vectors %s: %s", self._vectors_path, exc)
            return

        if len(ids) != len(arr):
            logger.warning(
                "ID count (%d) != vector count (%d); clearing index",
                len(ids),
                len(arr),
            )
            return

        self._ids = ids
        self._id_set = set(ids)
        self._vectors = arr.astype(np.float32) if len(arr) > 0 else None

    @staticmethod
    def _get_fastembed_version() -> str:
        """Return fastembed major.minor version, or ``"unknown"``."""
        try:
            from importlib.metadata import version as pkg_version
            full = pkg_version("fastembed")
            parts = full.split(".")
            return ".".join(parts[:2]) if len(parts) >= 2 else full
        except Exception:
            return "unknown"

    def _save(self) -> None:
        """Persist vectors and metadata atomically (tmp + replace)."""
        meta: dict[str, object] = {
            "ids": self._ids,
            "model_id": self._backend.model_id,
            "dim": self._backend.dim,
            "fastembed_version": self._get_fastembed_version(),
        }

        # -- atomic metadata write ----------------------------------------
        fd, tmp_meta = tempfile.mkstemp(dir=self._data_dir, suffix=".meta.tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_meta, str(self._meta_path))
            tmp_meta = ""  # mark as replaced
        finally:
            if tmp_meta:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_meta)

        # -- atomic vectors write -----------------------------------------
        if self._vectors is not None and len(self._vectors) > 0:
            fd2, tmp_vec = tempfile.mkstemp(dir=self._data_dir, suffix=".npy")
            os.close(fd2)
            try:
                np.save(tmp_vec, self._vectors)
                os.replace(tmp_vec, str(self._vectors_path))
                tmp_vec = ""
            finally:
                if tmp_vec:
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_vec)
        elif self._vectors_path.exists():
            self._vectors_path.unlink()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """L2-normalize along the last axis.  Zero vectors stay zero."""
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        result: npt.NDArray[np.floating] = v / norms
        return result

    # ------------------------------------------------------------------
    # LessonRetriever protocol
    # ------------------------------------------------------------------

    def add(self, lesson_id: str, text: str) -> None:
        """Add a single lesson embedding.  Idempotent (skips known IDs)."""
        if lesson_id in self._id_set:
            return

        # Embed outside the lock (potentially slow).
        raw = self._backend.embed([text])
        vec = self._normalize(np.array(raw[0], dtype=np.float32).reshape(1, -1))

        with self._lock:
            if lesson_id in self._id_set:  # double-check under lock
                return
            if self._vectors is None:
                self._vectors = vec
            else:
                self._vectors = np.vstack([self._vectors, vec])
            self._ids.append(lesson_id)
            self._id_set.add(lesson_id)
            self._save()

    def add_batch(self, items: list[tuple[str, str]]) -> None:
        """Add multiple (lesson_id, text) pairs in one embedding call.

        Skips IDs already in the index.  Faster than individual ``add``
        calls because embedding is batched.
        """
        new_items = [(lid, txt) for lid, txt in items if lid not in self._id_set]
        if not new_items:
            return

        texts = [txt for _, txt in new_items]
        raw = self._backend.embed(texts)
        arr = self._normalize(np.array(raw, dtype=np.float32))

        with self._lock:
            # Re-filter under lock (another thread may have added some).
            keep: list[int] = []
            keep_ids: list[str] = []
            for i, (lid, _) in enumerate(new_items):
                if lid not in self._id_set:
                    keep.append(i)
                    keep_ids.append(lid)
            if not keep:
                return

            new_arr = arr[keep]
            if self._vectors is None:
                self._vectors = new_arr
            else:
                self._vectors = np.vstack([self._vectors, new_arr])
            for lid in keep_ids:
                self._ids.append(lid)
                self._id_set.add(lid)
            self._save()

    def remove(self, lesson_id: str) -> None:
        """Remove a lesson from the index.  No-op if ID unknown."""
        with self._lock:
            if lesson_id not in self._id_set:
                return

            idx = self._ids.index(lesson_id)
            self._ids.pop(idx)
            self._id_set.discard(lesson_id)

            if self._vectors is not None:
                self._vectors = np.delete(self._vectors, idx, axis=0)
                if len(self._vectors) == 0:
                    self._vectors = None

            self._save()

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to *k* (lesson_id, cosine_score) pairs, best first.

        Brute-force dot product on L2-normalized vectors.
        """
        with self._lock:
            if self._vectors is None or len(self._vectors) == 0:
                return []
            vectors = self._vectors  # safe: vstack creates new arrays
            ids = list(self._ids)

        # Embed query outside lock.
        raw = self._backend.embed([query])
        qarr = self._normalize(np.array(raw[0], dtype=np.float32).reshape(1, -1))

        scores: npt.NDArray[np.floating] = (vectors @ qarr.T).flatten()
        top_k = min(k, len(scores))
        top_indices: npt.NDArray[np.intp] = np.argsort(scores)[::-1][:top_k]

        return [(ids[int(i)], float(scores[int(i)])) for i in top_indices]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of indexed lessons."""
        return len(self._ids)

    def __contains__(self, lesson_id: str) -> bool:
        """Check if a lesson ID is indexed."""
        return lesson_id in self._id_set
