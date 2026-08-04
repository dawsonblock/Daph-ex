"""Append-only JSONL stores with provenance and type invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schema import MemoryRecord, MemoryType


class JsonlMemoryStore:
    expected_type: MemoryType

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[MemoryRecord]:
        if not self.path.exists():
            return iter(())
        return iter(
            MemoryRecord.from_dict(json.loads(line))
            for line in self.path.read_text().splitlines() if line.strip()
        )

    def get(self, memory_id: str) -> MemoryRecord | None:
        return next((row for row in self if row.memory_id == memory_id), None)

    def append(self, record: MemoryRecord) -> None:
        if record.memory_type != self.expected_type:
            raise ValueError(f"Expected {self.expected_type.value}, got {record.memory_type.value}")
        existing = self.get(record.memory_id)
        if existing is not None:
            if existing != record:
                raise ValueError(f"Memory ID {record.memory_id!r} is immutable")
            return
        with self.path.open("a") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


class SourceMemoryStore(JsonlMemoryStore):
    expected_type = MemoryType.SOURCE


class SemanticMemoryStore(JsonlMemoryStore):
    expected_type = MemoryType.SEMANTIC


class EpisodicMemoryStore(JsonlMemoryStore):
    expected_type = MemoryType.EPISODIC


class ProceduralMemoryStore(JsonlMemoryStore):
    expected_type = MemoryType.PROCEDURAL


class ConsolidatedMemoryStore(JsonlMemoryStore):
    expected_type = MemoryType.CONSOLIDATED
