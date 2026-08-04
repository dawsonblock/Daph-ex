from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class MemoryType(str, Enum):
    SOURCE = "source"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class MemoryStatus(str, Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    memory_type: MemoryType
    content: str
    source_id: str
    source_span: str = ""
    confidence: float = 1.0
    status: MemoryStatus = MemoryStatus.CURRENT
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    embedding_version: str = "none"
    parent_id: str | None = None
    supersedes: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.memory_id or not self.source_id or not self.content.strip():
            raise ValueError("memory_id, source_id, and content are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        data["status"] = self.status.value
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryRecord":
        return cls(
            **{
                **dict(data),
                "memory_type": MemoryType(data["memory_type"]),
                "status": MemoryStatus(data.get("status", "current")),
                "tags": tuple(data.get("tags", ())),
            }
        )
