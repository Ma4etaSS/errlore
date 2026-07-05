"""Lesson subsystem: error tracking, lesson extraction, reinforcement, and decay.

Public API:
    ErrorRecord  -- dataclass for logged errors
    Lesson       -- dataclass for extracted lessons
    LessonStore  -- persistent store backed by JSONL via errlore.io
"""

from errlore.lessons.models import ErrorRecord, Lesson
from errlore.lessons.store import LessonStore

__all__ = [
    "ErrorRecord",
    "Lesson",
    "LessonStore",
]
