"""Fail-closed memory lifecycle transitions."""

from __future__ import annotations

from dataclasses import replace

from .schema import MemoryRecord, MemoryStatus, MemoryType


_TRANSITIONS = {
    MemoryStatus.RAW: {MemoryStatus.VERIFIED, MemoryStatus.REJECTED},
    MemoryStatus.CANDIDATE: {MemoryStatus.VERIFIED, MemoryStatus.REJECTED},
    MemoryStatus.VERIFIED: {MemoryStatus.PROMOTED, MemoryStatus.REJECTED},
    MemoryStatus.PROMOTED: {MemoryStatus.SUPERSEDED},
    MemoryStatus.SUPERSEDED: set(),
    MemoryStatus.REJECTED: set(),
}


class MemoryLifecycle:
    """Validate derived-memory promotion without mutating source records."""

    @staticmethod
    def transition(record: MemoryRecord, target: MemoryStatus) -> MemoryRecord:
        target = MemoryStatus(target)
        if record.memory_type == MemoryType.SOURCE:
            raise ValueError("Immutable source memory cannot transition")
        if target not in _TRANSITIONS[record.status]:
            raise ValueError(f"Illegal memory transition: {record.status.value}->{target.value}")
        return replace(record, status=target)

    @staticmethod
    def new_generated_candidate(
        *, memory_id: str, memory_type: MemoryType, content: str, source_id: str,
        **kwargs: object,
    ) -> MemoryRecord:
        if memory_type == MemoryType.SOURCE:
            raise ValueError("Generated statements cannot become source memory")
        return MemoryRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            source_id=source_id,
            status=MemoryStatus.CANDIDATE,
            **kwargs,
        )
