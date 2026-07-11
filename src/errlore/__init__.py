"""errlore -- memory for AI agents that learns from failures."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    # Single source of truth: installed package metadata (pyproject version).
    # A hardcoded string here shipped 0.2.x wheels reporting themselves 0.1.4.
    __version__ = _pkg_version("errlore")
except PackageNotFoundError:  # source tree without an install
    __version__ = "0.0.0.dev0"

from errlore.facade import AgentMemory, Injection
from errlore.lessons.store import LessonStore
from errlore.trust import FeedbackSignal, TrustEngine

__all__ = [
    "AgentMemory",
    "FeedbackSignal",
    "Injection",
    "LessonStore",
    "TrustEngine",
    "__version__",
]
