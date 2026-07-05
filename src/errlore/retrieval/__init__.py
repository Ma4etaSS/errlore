"""Semantic retrieval layer for lessons.

Provides embedding backends and a vector index for semantic search
over lessons.  Heavy dependencies (numpy, fastembed) live in submodules
and are imported lazily -- this ``__init__`` only defines lightweight
Protocol types that can be imported without extras.

Public API:
    LessonRetriever  -- structural Protocol consumed by LessonStore
    EmbeddingBackend -- structural Protocol for embedding providers

Concrete implementations (require ``errlore[embeddings]``):
    ``errlore.retrieval.backend.FastEmbedBackend``
    ``errlore.retrieval.backend.CallableBackend``
    ``errlore.retrieval.index.VectorIndex``
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LessonRetriever(Protocol):
    """Protocol for semantic retrieval backends used by LessonStore.

    Any object implementing ``search``, ``add``, and ``remove`` with
    matching signatures satisfies this protocol (structural subtyping).
    """

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to *k* (lesson_id, score) pairs ranked by similarity."""
        ...

    def add(self, lesson_id: str, text: str) -> None:
        """Index a lesson text under the given ID.  Idempotent."""
        ...

    def remove(self, lesson_id: str) -> None:
        """Remove a lesson from the index.  No-op if ID unknown."""
        ...


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Protocol for pluggable embedding providers."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...

    @property
    def model_id(self) -> str:
        """Unique model identifier (used for index compatibility checks)."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into float vectors."""
        ...


__all__ = [
    "EmbeddingBackend",
    "LessonRetriever",
]
