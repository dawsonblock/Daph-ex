"""Token-budgeted, redundancy-aware PrefixLM working-memory packets."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from hrm_adaptive_memory.memory.chunking import Chunk, approximate_tokens
from hrm_adaptive_memory.retrieval.hybrid import RetrievalCandidate


@dataclass(frozen=True)
class ContextBudget:
    total: int = 4096
    task: int = 500
    evidence: int = 2200
    state: int = 400
    generation: int = 996

    def __post_init__(self) -> None:
        if min(self.total, self.task, self.evidence, self.state, self.generation) < 0:
            raise ValueError("Context budgets cannot be negative")
        if self.task + self.evidence + self.state + self.generation > self.total:
            raise ValueError("Component budgets exceed total context")


@dataclass(frozen=True)
class EvidenceItem:
    chunk: Chunk
    relevance: float
    conflicting: bool = False


@dataclass(frozen=True)
class EvidencePacket:
    objective: str
    current_state: str
    evidence: tuple[EvidenceItem, ...]
    unresolved: tuple[str, ...]
    response_requirement: str
    rendered: str
    token_count: int
    evidence_tokens: int
    selected_chunk_ids: tuple[str, ...]
    provenance: Mapping[str, str] = field(default_factory=dict)


def _terms(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(left: str, right: str) -> float:
    a, b = _terms(left), _terms(right)
    return len(a & b) / max(1, len(a | b))


class EvidencePacker:
    def __init__(self, *, budget: ContextBudget | None = None,
                 token_counter: Callable[[str], int] = approximate_tokens,
                 redundancy_weight: float = 0.25):
        self.budget = budget or ContextBudget(); self.token_counter = token_counter
        self.redundancy_weight = redundancy_weight

    def select(self, candidates: Sequence[RetrievalCandidate | EvidenceItem]) -> tuple[EvidenceItem, ...]:
        remaining = [
            item if isinstance(item, EvidenceItem) else EvidenceItem(
                item.chunk, item.reranker_score if item.reranker_score is not None else item.rrf_score,
            ) for item in candidates
        ]
        selected: list[EvidenceItem] = []; tokens = 0
        while remaining:
            scored = []
            for item in remaining:
                redundancy = max((jaccard(item.chunk.content, old.chunk.content) for old in selected), default=0.0)
                value = item.relevance - self.redundancy_weight * redundancy
                scored.append((value, -item.chunk.token_count, item.chunk.chunk_id, item))
            item = max(scored, key=lambda row: row[:3])[3]
            remaining.remove(item)
            if tokens + item.chunk.token_count > self.budget.evidence:
                continue
            selected.append(item); tokens += item.chunk.token_count
        return tuple(selected)

    def pack(self, *, objective: str, current_state: str,
             candidates: Sequence[RetrievalCandidate | EvidenceItem],
             unresolved: Iterable[str] = (),
             response_requirement: str = "Answer using supplied evidence; identify missing evidence.") -> EvidencePacket:
        selected = self.select(candidates)
        normal = [item for item in selected if not item.conflicting]
        conflicts = [item for item in selected if item.conflicting]
        parts = ["[OBJECTIVE]", objective, "[CURRENT STATE]", current_state, "[HIGH-CONFIDENCE EVIDENCE]"]
        for index, item in enumerate(normal, 1):
            parts.extend([f"[E{index}] source: {item.chunk.source_id} section: {item.chunk.section} relevance: {item.relevance:.4f}", item.chunk.content])
        if conflicts:
            parts.append("[CONFLICTING EVIDENCE]")
            for index, item in enumerate(conflicts, len(normal) + 1):
                parts.extend([f"[E{index}] source: {item.chunk.source_id} relevance: {item.relevance:.4f}", item.chunk.content])
        unresolved_values = tuple(unresolved)
        parts.append("[UNRESOLVED INFORMATION]")
        parts.extend(f"- {value}" for value in unresolved_values)
        parts.extend(["[RESPONSE REQUIREMENT]", response_requirement])
        rendered = "\n".join(parts)
        count = self.token_counter(rendered)
        prompt_limit = self.budget.total - self.budget.generation
        if count > prompt_limit:
            raise ValueError(f"Packed prompt uses {count} tokens, above limit {prompt_limit}")
        return EvidencePacket(
            objective=objective, current_state=current_state, evidence=selected,
            unresolved=unresolved_values, response_requirement=response_requirement,
            rendered=rendered, token_count=count,
            evidence_tokens=sum(item.chunk.token_count for item in selected),
            selected_chunk_ids=tuple(item.chunk.chunk_id for item in selected),
            provenance={item.chunk.chunk_id: item.chunk.source_id for item in selected},
        )
