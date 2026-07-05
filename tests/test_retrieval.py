"""Tests for errlore.retrieval (embedding backends, vector index, store integration).

The bulk of the tests use a deterministic fake backend so they run in CI
without fastembed.  A small smoke-test group uses real fastembed and is
skipped when the extras are not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from errlore.lessons.store import LessonStore
from errlore.retrieval import EmbeddingBackend, LessonRetriever
from errlore.retrieval.backend import CallableBackend
from errlore.retrieval.index import VectorIndex

# ------------------------------------------------------------------
# Deterministic fake backend
# ------------------------------------------------------------------

# Pre-defined 4-dim embeddings for known texts.
# Vectors are designed so that "api timeout" is close to "service delay"
# and far from "json parsing".
_KNOWN_EMBEDDINGS: dict[str, list[float]] = {
    "api timeout": [1.0, 0.0, 0.0, 0.0],
    "service delay": [0.9, 0.1, 0.0, 0.0],
    "json parsing": [0.0, 1.0, 0.0, 0.0],
    "database lock": [0.0, 0.0, 1.0, 0.0],
    "query: slow api": [0.95, 0.05, 0.0, 0.0],
    "query: parse json": [0.05, 0.95, 0.0, 0.0],
    "query: db stuck": [0.0, 0.05, 0.95, 0.0],
}


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return pre-defined or hash-based embeddings."""
    results: list[list[float]] = []
    for t in texts:
        if t in _KNOWN_EMBEDDINGS:
            results.append(_KNOWN_EMBEDDINGS[t])
        else:
            # Deterministic fallback: hash-based
            h = hash(t) & 0xFFFFFFFF
            vec = [
                float((h >> 0) & 0xFF) / 255,
                float((h >> 8) & 0xFF) / 255,
                float((h >> 16) & 0xFF) / 255,
                float((h >> 24) & 0xFF) / 255,
            ]
            results.append(vec)
    return results


@pytest.fixture()
def fake_backend() -> CallableBackend:
    """Deterministic 4-dim backend for unit tests."""
    return CallableBackend(fn=_fake_embed, dim=4, model_id="test-v1")


@pytest.fixture()
def index(data_dir: Path, fake_backend: CallableBackend) -> VectorIndex:
    """VectorIndex backed by the fake backend in a tmp directory."""
    return VectorIndex(data_dir, fake_backend)


# ==================================================================
# CallableBackend
# ==================================================================


class TestCallableBackend:
    """CallableBackend wraps any callable as EmbeddingBackend."""

    def test_properties(self, fake_backend: CallableBackend) -> None:
        assert fake_backend.dim == 4
        assert fake_backend.model_id == "test-v1"

    def test_embed(self, fake_backend: CallableBackend) -> None:
        result = fake_backend.embed(["api timeout", "json parsing"])
        assert len(result) == 2
        assert len(result[0]) == 4
        assert result[0] == [1.0, 0.0, 0.0, 0.0]

    def test_satisfies_protocol(self, fake_backend: CallableBackend) -> None:
        assert isinstance(fake_backend, EmbeddingBackend)


# ==================================================================
# VectorIndex
# ==================================================================


class TestVectorIndexAdd:
    """add / __len__ / __contains__."""

    def test_add_single(self, index: VectorIndex) -> None:
        assert len(index) == 0
        index.add("L1", "api timeout")
        assert len(index) == 1
        assert "L1" in index

    def test_add_idempotent(self, index: VectorIndex) -> None:
        index.add("L1", "api timeout")
        index.add("L1", "api timeout")
        assert len(index) == 1

    def test_add_batch(self, index: VectorIndex) -> None:
        index.add_batch([("L1", "api timeout"), ("L2", "json parsing")])
        assert len(index) == 2
        assert "L1" in index
        assert "L2" in index

    def test_add_batch_skips_existing(self, index: VectorIndex) -> None:
        index.add("L1", "api timeout")
        index.add_batch([("L1", "api timeout"), ("L2", "json parsing")])
        assert len(index) == 2


class TestVectorIndexSearch:
    """search returns correctly ranked results."""

    def test_search_empty(self, index: VectorIndex) -> None:
        assert index.search("anything", k=3) == []

    def test_search_ranking(self, index: VectorIndex) -> None:
        index.add("L1", "api timeout")
        index.add("L2", "json parsing")
        index.add("L3", "database lock")

        results = index.search("query: slow api", k=3)
        ids = [r[0] for r in results]
        scores = [r[1] for r in results]

        # L1 (api timeout) should be most similar to "query: slow api"
        assert ids[0] == "L1"
        # Scores should be descending
        assert scores[0] >= scores[1] >= scores[2]

    def test_search_respects_k(self, index: VectorIndex) -> None:
        index.add_batch([
            ("L1", "api timeout"),
            ("L2", "json parsing"),
            ("L3", "database lock"),
        ])
        results = index.search("query: slow api", k=1)
        assert len(results) == 1

    def test_search_cosine_score_range(self, index: VectorIndex) -> None:
        index.add("L1", "api timeout")
        results = index.search("api timeout", k=1)
        # Same text -> cosine ~ 1.0
        assert results[0][1] > 0.99

    def test_satisfies_protocol(self, index: VectorIndex) -> None:
        assert isinstance(index, LessonRetriever)


class TestVectorIndexRemove:
    """remove drops the entry from index and disk."""

    def test_remove_existing(self, index: VectorIndex) -> None:
        index.add("L1", "api timeout")
        index.add("L2", "json parsing")
        index.remove("L1")
        assert len(index) == 1
        assert "L1" not in index
        assert "L2" in index

    def test_remove_nonexistent(self, index: VectorIndex) -> None:
        index.add("L1", "api timeout")
        index.remove("NOPE")  # no-op
        assert len(index) == 1

    def test_remove_last(self, index: VectorIndex) -> None:
        index.add("L1", "api timeout")
        index.remove("L1")
        assert len(index) == 0
        # Search on empty should return empty
        assert index.search("anything") == []


class TestVectorIndexPersist:
    """Persisted index survives re-instantiation."""

    def test_persist_roundtrip(
        self, data_dir: Path, fake_backend: CallableBackend,
    ) -> None:
        idx1 = VectorIndex(data_dir, fake_backend)
        idx1.add("L1", "api timeout")
        idx1.add("L2", "json parsing")

        # Re-open from same directory
        idx2 = VectorIndex(data_dir, fake_backend)
        assert len(idx2) == 2
        assert "L1" in idx2
        assert "L2" in idx2

        # Search should still work
        results = idx2.search("query: slow api", k=1)
        assert results[0][0] == "L1"

    def test_persist_after_remove(
        self, data_dir: Path, fake_backend: CallableBackend,
    ) -> None:
        idx1 = VectorIndex(data_dir, fake_backend)
        idx1.add("L1", "api timeout")
        idx1.add("L2", "json parsing")
        idx1.remove("L1")

        idx2 = VectorIndex(data_dir, fake_backend)
        assert len(idx2) == 1
        assert "L1" not in idx2


class TestVectorIndexCorruptLoad:
    """B5: corrupted on-disk files do not crash the constructor."""

    def test_corrupt_vectors_npy(self, data_dir: Path, fake_backend: CallableBackend) -> None:
        """Corrupted vectors.npy -> empty index, no exception."""
        # Write a valid index first.
        idx1 = VectorIndex(data_dir, fake_backend)
        idx1.add("L1", "api timeout")
        assert len(idx1) == 1

        # Corrupt vectors.npy
        vectors_path = data_dir / "vectors.npy"
        vectors_path.write_bytes(b"this is not a numpy file")

        # Re-open: should not raise, just produce empty index.
        idx2 = VectorIndex(data_dir, fake_backend)
        assert len(idx2) == 0

    def test_corrupt_meta_json(self, data_dir: Path, fake_backend: CallableBackend) -> None:
        """Corrupted vector_meta.json -> empty index, no exception."""
        idx1 = VectorIndex(data_dir, fake_backend)
        idx1.add("L1", "api timeout")

        meta_path = data_dir / "vector_meta.json"
        meta_path.write_text("{invalid json", encoding="utf-8")

        idx2 = VectorIndex(data_dir, fake_backend)
        assert len(idx2) == 0


class TestVectorIndexReindex:
    """Model change triggers reindex."""

    def test_model_change_clears_index(self, data_dir: Path) -> None:
        backend_v1 = CallableBackend(fn=_fake_embed, dim=4, model_id="model-v1")
        idx1 = VectorIndex(data_dir, backend_v1)
        idx1.add("L1", "api timeout")
        assert len(idx1) == 1

        # Open with different model_id
        backend_v2 = CallableBackend(fn=_fake_embed, dim=4, model_id="model-v2")
        idx2 = VectorIndex(data_dir, backend_v2)
        # Old vectors should be discarded
        assert len(idx2) == 0
        assert "L1" not in idx2


# ==================================================================
# LessonStore with semantic retriever
# ==================================================================


class _FakeRetriever:
    """Minimal retriever that returns hardcoded results."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.search_results: list[tuple[str, float]] = []

    def add(self, lesson_id: str, text: str) -> None:
        self._store[lesson_id] = text

    def remove(self, lesson_id: str) -> None:
        self._store.pop(lesson_id, None)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search_results[:k]


class TestLessonStoreWithRetriever:
    """LessonStore uses the retriever for search_lessons when available."""

    def test_semantic_results_used(self, data_dir: Path) -> None:
        retriever = _FakeRetriever()
        store = LessonStore(data_dir, retriever=retriever)

        lid = store.log_lesson(pattern="api timeout handling", solution="use circuit breaker")
        assert lid in retriever._store  # lesson was added to retriever

        # Configure retriever to return the lesson
        retriever.search_results = [(lid, 0.95)]

        # Query that has ZERO word overlap with pattern
        results = store.search_lessons(query="external service hangs", limit=5)
        assert len(results) == 1
        assert results[0].id == lid

    def test_fallback_to_word_overlap(self, data_dir: Path) -> None:
        retriever = _FakeRetriever()
        store = LessonStore(data_dir, retriever=retriever)

        store.log_lesson(pattern="handle timeout errors", solution="retry")

        # Configure retriever to return nothing (semantic miss)
        retriever.search_results = []

        # Query that has word overlap with pattern
        results = store.search_lessons(query="timeout errors handling", limit=5)
        assert len(results) >= 1
        assert "timeout" in results[0].pattern.lower()

    def test_no_retriever_uses_word_overlap(self, data_dir: Path) -> None:
        store = LessonStore(data_dir)  # no retriever
        store.log_lesson(pattern="handle timeout errors", solution="retry")
        results = store.search_lessons(query="timeout errors handling", limit=5)
        assert len(results) >= 1

    def test_structured_filters_with_retriever(self, data_dir: Path) -> None:
        retriever = _FakeRetriever()
        store = LessonStore(data_dir, retriever=retriever)

        lid1 = store.log_lesson(
            pattern="api stall", solution="s1", task_type="infra",
        )
        lid2 = store.log_lesson(
            pattern="json break", solution="s2", task_type="code",
        )

        # Retriever returns both but filter narrows to task_type=infra
        retriever.search_results = [(lid1, 0.9), (lid2, 0.8)]

        results = store.search_lessons(
            query="service down", task_type="infra", limit=5,
        )
        assert len(results) == 1
        assert results[0].id == lid1

    def test_retriever_receives_add_on_log(self, data_dir: Path) -> None:
        retriever = _FakeRetriever()
        store = LessonStore(data_dir, retriever=retriever)

        lid = store.log_lesson(pattern="new lesson", solution="new solution")
        assert lid in retriever._store
        assert retriever._store[lid] == "new lesson"

    def test_sync_existing_lessons(self, data_dir: Path) -> None:
        # Create store without retriever, add lessons
        store1 = LessonStore(data_dir)
        lid1 = store1.log_lesson(pattern="existing lesson one", solution="s1")
        lid2 = store1.log_lesson(pattern="existing lesson two", solution="s2")

        # Re-open with retriever -- should sync
        retriever = _FakeRetriever()
        store2 = LessonStore(data_dir, retriever=retriever)
        assert lid1 in retriever._store
        assert lid2 in retriever._store

        # Suppress unused variable warning
        assert store2 is not None


class TestLessonStoreIntegrationWithVectorIndex:
    """LessonStore + VectorIndex end-to-end (uses fake backend, no fastembed)."""

    def test_end_to_end_search(self, data_dir: Path, fake_backend: CallableBackend) -> None:
        index = VectorIndex(data_dir, fake_backend)
        store = LessonStore(data_dir, retriever=index)

        store.log_lesson(pattern="api timeout", solution="add circuit breaker")
        store.log_lesson(pattern="json parsing", solution="strip fences")
        store.log_lesson(pattern="database lock", solution="use WAL")

        results = store.search_lessons(query="query: slow api", limit=2)
        assert len(results) >= 1
        assert results[0].pattern == "api timeout"


# ==================================================================
# Real fastembed tests (skipped if not installed)
# ==================================================================

_has_fastembed = True
try:
    import fastembed  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    _has_fastembed = False


@pytest.mark.skipif(not _has_fastembed, reason="fastembed not installed")
class TestFastEmbedSmoke:
    """Smoke tests with real fastembed model."""

    def test_embed_english(self) -> None:
        from errlore.retrieval.backend import FastEmbedBackend

        backend = FastEmbedBackend()
        result = backend.embed(["API timeout error handling"])
        assert len(result) == 1
        assert len(result[0]) == backend.dim
        assert backend.dim == 384

    def test_embed_russian(self) -> None:
        from errlore.retrieval.backend import FastEmbedBackend

        backend = FastEmbedBackend()
        result = backend.embed(["Обработка ошибок таймаута API"])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_search_multilingual(self, data_dir: Path) -> None:
        from errlore.retrieval.backend import FastEmbedBackend

        backend = FastEmbedBackend()
        index = VectorIndex(data_dir, backend)

        index.add("L1", "API request timeout due to server overload")
        index.add("L2", "JSON parsing fails on malformed input")
        index.add("L3", "Database connection pool exhaustion")

        # English query
        results = index.search("service hangs when the server is busy", k=1)
        assert results[0][0] == "L1"

    def test_search_russian(self, data_dir: Path) -> None:
        from errlore.retrieval.backend import FastEmbedBackend

        backend = FastEmbedBackend()
        index = VectorIndex(data_dir, backend)

        index.add("L1", "Запрос к API зависает из-за перегрузки сервера")
        index.add("L2", "Парсинг JSON падает на некорректном вводе")
        index.add("L3", "Пул соединений к базе данных исчерпан")

        # Russian query
        results = index.search("сервис не отвечает при высокой нагрузке", k=1)
        assert results[0][0] == "L1"

    def test_fastembed_backend_protocol(self) -> None:
        from errlore.retrieval.backend import FastEmbedBackend

        backend = FastEmbedBackend()
        assert isinstance(backend, EmbeddingBackend)
