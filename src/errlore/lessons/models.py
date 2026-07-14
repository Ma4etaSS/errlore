"""Data models for the lesson subsystem."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _short_id() -> str:
    """Generate a 12-char hex ID from uuid4."""
    return uuid.uuid4().hex[:12]


def _utc_now_iso() -> str:
    """Return timezone-aware UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ErrorRecord:
    """A recorded error event."""

    model: str
    task_type: str
    error_type: str
    message: str
    id: str = field(default_factory=_short_id)
    timestamp: str = field(default_factory=_utc_now_iso)
    resolved: bool = False
    resolution: str = ""
    context: str = ""
    stacktrace: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSONL storage."""
        d: dict[str, Any] = {
            "id": self.id,
            "model": self.model,
            "task_type": self.task_type,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "context": self.context,
            "stacktrace": self.stacktrace,
        }
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ErrorRecord:
        """Deserialize from a JSONL dict."""
        return cls(
            id=str(d.get("id", _short_id())),
            model=str(d.get("model", "")),
            task_type=str(d.get("task_type", "")),
            error_type=str(d.get("error_type", "")),
            message=str(d.get("message", "")),
            timestamp=str(d.get("timestamp", _utc_now_iso())),
            resolved=bool(d.get("resolved", False)),
            resolution=str(d.get("resolution", "")),
            context=str(d.get("context", "")),
            stacktrace=str(d.get("stacktrace", "")),
            metadata=d.get("metadata"),
        )


@dataclass
class Lesson:
    """An extracted lesson linking a problem pattern to its solution."""

    pattern: str
    solution: str
    id: str = field(default_factory=_short_id)
    task_type: str = ""
    error_type: str = ""
    confidence: float = 0.8
    applied_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    source_error_id: str = ""
    source_errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSONL storage."""
        d: dict[str, Any] = {
            "id": self.id,
            "pattern": self.pattern,
            "solution": self.solution,
            "task_type": self.task_type,
            "error_type": self.error_type,
            "confidence": self.confidence,
            "applied_count": self.applied_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_error_id": self.source_error_id,
            "source_errors": self.source_errors,
        }
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Lesson:
        """Deserialize from a JSONL dict."""
        raw_source_errors = d.get("source_errors")
        source_errors: list[str] = (
            [str(e) for e in raw_source_errors] if isinstance(raw_source_errors, list) else []
        )
        return cls(
            id=str(d.get("id", _short_id())),
            pattern=str(d.get("pattern", "")),
            solution=str(d.get("solution", "")),
            task_type=str(d.get("task_type", "")),
            error_type=str(d.get("error_type", "")),
            confidence=float(d.get("confidence", 0.8)),
            applied_count=int(d.get("applied_count", 0)),
            success_count=int(d.get("success_count", 0)),
            failure_count=int(d.get("failure_count", 0)),
            created_at=str(d.get("created_at", _utc_now_iso())),
            updated_at=str(d.get("updated_at", _utc_now_iso())),
            source_error_id=str(d.get("source_error_id", "")),
            source_errors=source_errors,
            metadata=d.get("metadata"),
        )
