"""Evidence selection under a token budget, with near-duplicate suppression.

Gate B measured that retrieval *precision* is a binding constraint: with the
required evidence always present, quality fell from 1.00 with unrelated
padding to 0.39 with the retriever's own top-k, because near-duplicate records
differing only in their entity identifiers confuse the reader
(`evidence/gate_b/qualification/packing_diagnostic.json`). Selection therefore has to
suppress look-alikes, not merely fit a budget.

Two complementary filters:

  entity anchoring   drop records that share no entity with the question or an
                     accepted bridge entity — the direct answer to the Gate B
                     confusability finding
  MMR redundancy     among the survivors, penalise records that duplicate
                     material already selected
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ..retrieval.lexical import tokenize
from .state import EvidenceRecordView


@dataclass(frozen=True)
class SelectionReceipt:
    selected_ids: tuple[str, ...]
    dropped_unanchored_ids: tuple[str, ...]
    dropped_redundant_ids: tuple[str, ...]
    dropped_over_budget_ids: tuple[str, ...]
    selected_tokens: int
    lambda_redundancy: float
    anchor_entities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def select_evidence(
    records: Sequence[EvidenceRecordView], *,
    anchor_entities: Sequence[str] = (),
    token_budget: int = 2200,
    max_records: int | None = None,
    lambda_redundancy: float = 0.5,
    enforce_anchoring: bool = True,
) -> tuple[tuple[EvidenceRecordView, ...], SelectionReceipt]:
    """Select a low-redundancy, entity-anchored subset within the token budget.

    Records are considered in retrieval order, so a record's relevance is its
    rank; MMR then discounts each candidate by its similarity to what has
    already been selected.
    """

    anchors = {value for value in anchor_entities}
    unanchored: list[str] = []
    candidates: list[EvidenceRecordView] = []
    for record in records:
        if enforce_anchoring and anchors and not (set(record.entities) & anchors):
            unanchored.append(record.evidence_id)
            continue
        candidates.append(record)

    selected: list[EvidenceRecordView] = []
    selected_tokens = 0
    redundant: list[str] = []
    over_budget: list[str] = []
    seen_ids: set[str] = set()
    token_sets: list[set[str]] = []

    for record in candidates:
        if record.evidence_id in seen_ids:
            redundant.append(record.evidence_id)
            continue
        terms = set(tokenize(record.content))
        similarity = max((_jaccard(terms, other) for other in token_sets), default=0.0)
        # Rank-based relevance in [0, 1]; later ranks are worth less.
        relevance = 1.0 / (1 + max(0, record.rank - 1))
        if token_sets and relevance - lambda_redundancy * similarity <= 0.0:
            redundant.append(record.evidence_id)
            continue
        if max_records is not None and len(selected) >= max_records:
            over_budget.append(record.evidence_id)
            continue
        if selected_tokens + record.token_count > token_budget:
            over_budget.append(record.evidence_id)
            continue
        selected.append(record)
        seen_ids.add(record.evidence_id)
        token_sets.append(terms)
        selected_tokens += record.token_count

    receipt = SelectionReceipt(
        selected_ids=tuple(row.evidence_id for row in selected),
        dropped_unanchored_ids=tuple(unanchored),
        dropped_redundant_ids=tuple(redundant),
        dropped_over_budget_ids=tuple(over_budget),
        selected_tokens=selected_tokens,
        lambda_redundancy=lambda_redundancy,
        anchor_entities=tuple(sorted(anchors)),
    )
    return tuple(selected), receipt


CAPABILITY_USE_REQUIREMENT = (
    "Return only the answer. Use supplied evidence when helpful, "
    "but answer the task regardless."
)


def compose_evidence_prompt(
    question: str, contents: Sequence[str], *,
    response_requirement: str = CAPABILITY_USE_REQUIREMENT,
) -> str:
    """Compose the model-visible prompt.

    Byte-identical to the Gate A composer
    (`experiments.context_study.ContextConstructor._compose`) so that arms
    measured here remain comparable to the frozen Gate A and Gate B numbers.
    A parity test pins the two together.
    """

    parts = ["[OBJECTIVE]", question, "[EVIDENCE]"]
    if not contents:
        parts.append("[NO EXTERNAL EVIDENCE]")
    for index, content in enumerate(contents, 1):
        parts.extend([f"[E{index}]", content])
    parts.extend(["[RESPONSE REQUIREMENT]", response_requirement])
    return "\n".join(parts)
