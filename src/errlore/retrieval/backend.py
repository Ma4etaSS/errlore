"""Concrete embedding backend implementations.

``FastEmbedBackend`` -- multilingual embeddings via fastembed (lazy import).
``CallableBackend`` -- wrap any user-supplied function (ollama, openai, etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Default: 384-dim, 50+ languages (RU + EN), 512 tokens, fast.
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedBackend:
    """Multilingual embedding backend powered by fastembed.

    The model is loaded lazily on first ``embed`` call.  If fastembed
    is not installed, a clear ``ImportError`` is raised with install
    instructions.

    Args:
        model_name: fastembed model identifier.  Must appear in
            ``TextEmbedding.list_supported_models()``.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dim: int | None = None

    def _ensure_model(self) -> None:
        """Lazy-load the fastembed model on first use."""
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise ImportError(
                "fastembed is required for embedding-based retrieval.  "
                "Install it with:  pip install errlore[embeddings]"
            ) from None
        self._model = TextEmbedding(self._model_name)
        # Probe dimensionality.
        probe = list(self._model.embed(["_dim_probe_"]))
        self._dim = len(probe[0])

    # -- EmbeddingBackend protocol -----------------------------------------

    @property
    def dim(self) -> int:
        """Embedding dimensionality (discovered on first use)."""
        self._ensure_model()
        assert self._dim is not None
        return self._dim

    @property
    def model_id(self) -> str:
        """Model identifier string."""
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into float vectors via fastembed."""
        self._ensure_model()
        assert self._model is not None
        raw = list(self._model.embed(texts))
        return [vec.tolist() for vec in raw]


class CallableBackend:
    """Wrap an arbitrary callable as an ``EmbeddingBackend``.

    Useful for plugging in OpenAI, Ollama, or any custom embedding
    function without writing a full backend class.

    Args:
        fn: Callable that takes ``list[str]`` and returns
            ``list[list[float]]``.
        dim: Fixed embedding dimensionality.
        model_id: Identifier string (used for index compatibility).
    """

    def __init__(
        self,
        fn: Callable[[list[str]], list[list[float]]],
        dim: int,
        model_id: str = "custom",
    ) -> None:
        self._fn = fn
        self._dim = dim
        self._model_id = model_id

    @property
    def dim(self) -> int:
        """Fixed embedding dimensionality."""
        return self._dim

    @property
    def model_id(self) -> str:
        """Model identifier string."""
        return self._model_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Delegate to the wrapped callable."""
        return self._fn(texts)
