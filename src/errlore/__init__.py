"""errlore -- memory for AI agents that learns from failures."""

__version__ = "0.1.2"

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
