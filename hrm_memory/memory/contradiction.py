from __future__ import annotations

from dataclasses import dataclass

from .schema import MemoryRecord, MemoryStatus


@dataclass(frozen=True)
class LineageEdge:
    prior_id: str
    current_id: str
    relation: str


class ContradictionLedger:
    def __init__(self) -> None:
        self._edges: list[LineageEdge] = []

    def supersede(self, prior: MemoryRecord, current: MemoryRecord) -> LineageEdge:
        if current.supersedes != prior.memory_id:
            raise ValueError("Current record must name the prior record in supersedes")
        edge = LineageEdge(prior.memory_id, current.memory_id, MemoryStatus.SUPERSEDED.value)
        self._edges.append(edge)
        return edge

    def lineage(self, memory_id: str) -> tuple[LineageEdge, ...]:
        return tuple(edge for edge in self._edges if memory_id in (edge.prior_id, edge.current_id))
