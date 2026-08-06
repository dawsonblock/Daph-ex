"""Structured evidence state.

Evidence sufficiency is never reduced to one unexplained scalar. The state
records what the question demands, what the retrieved records actually supply,
and precisely what is missing, so a follow-up action can be derived from the
gap rather than guessed from a confidence number.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

# Identifiers of the form Adapter-78103, Trial-000-867, Plan-000-965. Entity
# shape is a corpus property; natural-document corpora (Stage 7) must supply
# their own extractor rather than inherit this one.
ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z]*(?:-\d+)+\b")
NUMBER_PATTERN = re.compile(r"(?<![\w-])[-+]?\d+(?:\.\d+)?(?![\w-])")


def extract_entities(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ENTITY_PATTERN.findall(text)))


def extract_numbers(text: str) -> tuple[str, ...]:
    return tuple(NUMBER_PATTERN.findall(text))


@dataclass(frozen=True)
class EvidenceRecordView:
    """A retrieved record plus the structure extracted from it."""

    evidence_id: str
    source_id: str
    content: str
    token_count: int
    rank: int
    entities: tuple[str, ...]
    numbers: tuple[str, ...]

    @classmethod
    def from_retrieved(cls, row: Any) -> "EvidenceRecordView":
        content = str(row.content)
        return cls(
            evidence_id=str(row.evidence_id),
            source_id=str(row.source_id),
            content=content,
            token_count=int(getattr(row, "token_count", 0) or 0),
            rank=int(getattr(row, "rank", 0) or 0),
            entities=extract_entities(content),
            numbers=extract_numbers(content),
        )


@dataclass(frozen=True)
class EvidenceState:
    """What the question needs, what the evidence supplies, and what is absent."""

    question: str
    retrieved_ids: tuple[str, ...]
    required_entities: tuple[str, ...]
    observed_entities: tuple[str, ...]
    missing_entities: tuple[str, ...]
    bridge_entities: tuple[str, ...]
    linked_entities: tuple[str, ...]
    required_operands: tuple[str, ...]
    observed_operands: tuple[str, ...]
    answer_bearing_ids: tuple[str, ...]
    contradictions: tuple[Mapping[str, Any], ...]
    source_metadata: Mapping[str, Any]
    entity_coverage: float
    redundancy: float
    token_cost: int
    records: tuple[EvidenceRecordView, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("records", None)
        return row


def _support_counts(records: Sequence[EvidenceRecordView]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for entity in record.entities:
            counts[entity] = counts.get(entity, 0) + 1
    return counts


def _detect_contradictions(
    records: Sequence[EvidenceRecordView],
) -> tuple[Mapping[str, Any], ...]:
    """Flag entities bound to different values by different records.

    This is a signal to surface, not an automatic rejection: temporal updates
    legitimately supersede earlier values, and that distinction is Stage 9's
    problem, not this module's.
    """

    bindings: dict[str, set[str]] = {}
    holders: dict[str, list[str]] = {}
    for record in records:
        for entity in record.entities:
            for number in record.numbers:
                bindings.setdefault(entity, set()).add(number)
                holders.setdefault(entity, []).append(record.evidence_id)
    return tuple(
        {
            "entity": entity,
            "values": sorted(values),
            "evidence_ids": sorted(set(holders[entity])),
        }
        for entity, values in sorted(bindings.items())
        if len(values) > 1
    )


def build_evidence_state(
    *, question: str, records: Sequence[Any],
    required_operand_count: int = 0,
) -> EvidenceState:
    """Derive structured state from a question and its retrieved records."""

    views = tuple(
        row if isinstance(row, EvidenceRecordView) else EvidenceRecordView.from_retrieved(row)
        for row in records
    )
    question_entities = extract_entities(question)
    observed = tuple(dict.fromkeys(
        entity for view in views for entity in view.entities
    ))
    support = _support_counts(views)

    missing = tuple(entity for entity in question_entities if entity not in observed)
    # A bridge entity is (a) introduced by the evidence rather than the
    # question, (b) linked to the question's subject by co-occurring with it in
    # some record, and (c) still unresolved, appearing in only that one record.
    # Condition (b) is what distinguishes the second hop of the question from
    # the many unrelated entities a retriever also returns.
    question_set = set(question_entities)
    linked: set[str] = set()
    for view in views:
        if question_set & set(view.entities):
            linked |= set(view.entities) - question_set
    bridges = tuple(
        entity for entity in observed
        if entity in linked and support.get(entity, 0) < 2
    )
    # Linked entities stay relevant even once they are resolved: the record
    # that resolves a link mentions the link, not the question's subject, so
    # anchoring on question entities alone would discard the second hop.
    linked_entities = tuple(entity for entity in observed if entity in linked)
    operands = tuple(number for view in views for number in view.numbers)
    unique_ids = list(dict.fromkeys(view.evidence_id for view in views))
    # A record that mentions something the question asked about *and* states a
    # value is a candidate answer. Its presence is what distinguishes "the link
    # is unresolved" from "an unrelated link happens to be dangling".
    answer_bearing = tuple(
        view.evidence_id for view in views
        if (question_set & set(view.entities)) and view.numbers
    )

    return EvidenceState(
        question=question,
        retrieved_ids=tuple(view.evidence_id for view in views),
        required_entities=question_entities,
        observed_entities=observed,
        missing_entities=missing,
        bridge_entities=bridges,
        linked_entities=linked_entities,
        required_operands=tuple(str(index) for index in range(required_operand_count)),
        observed_operands=operands,
        answer_bearing_ids=answer_bearing,
        contradictions=_detect_contradictions(views),
        source_metadata={
            "record_count": len(views),
            "unique_record_count": len(unique_ids),
            "source_ids": sorted({view.source_id for view in views}),
        },
        entity_coverage=(
            1.0 if not question_entities
            else len([e for e in question_entities if e in observed]) / len(question_entities)
        ),
        redundancy=1.0 - len(unique_ids) / max(1, len(views)),
        token_cost=sum(view.token_count for view in views),
        records=views,
    )
