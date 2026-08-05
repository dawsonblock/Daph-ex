"""Deterministic evidence sufficiency and missing-information extraction.

The verdict names the specific gap and therefore the specific next action.
"Insufficient" without a named gap would give a controller nothing to act on.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from .state import EvidenceState

# Operation words that indicate the answer is derived rather than stated. This
# is a template-bounded heuristic over the controlled corpus; natural-document
# work (Stage 7) must re-validate or replace it.
_OPERATIONS = {
    "multiplies": "*", "multiplied": "*", "times": "*", "product": "*",
    "plus": "+", "sum": "+", "added": "+", "total of": "+",
    "minus": "-", "less": "-", "difference": "-",
    "divided": "/", "per": "/",
}


class SufficiencyVerdict(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    MISSING_BRIDGE = "MISSING_BRIDGE"
    MISSING_SUBJECT = "MISSING_SUBJECT"
    NEEDS_CALCULATION = "NEEDS_CALCULATION"
    CONFLICTING = "CONFLICTING"
    EMPTY = "EMPTY"


@dataclass(frozen=True)
class SufficiencyReport:
    verdict: SufficiencyVerdict
    missing_information: tuple[str, ...]
    followup_terms: tuple[str, ...]
    rationale: str
    detected_operation: str | None = None

    @property
    def needs_followup(self) -> bool:
        return self.verdict in (
            SufficiencyVerdict.MISSING_BRIDGE, SufficiencyVerdict.MISSING_SUBJECT,
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["verdict"] = self.verdict.value
        row["needs_followup"] = self.needs_followup
        return row


def detect_operation(state: EvidenceState) -> str | None:
    """Return an arithmetic operator when the evidence states a derivation rule."""

    text = " ".join(record.content.lower() for record in state.records)
    for word, operator in _OPERATIONS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return operator
    return None


def assess(state: EvidenceState, *, require_calculation: bool | None = None) -> SufficiencyReport:
    """Classify the evidence gap and name the follow-up terms that would close it."""

    if not state.records:
        return SufficiencyReport(
            SufficiencyVerdict.EMPTY, ("no evidence retrieved",),
            state.required_entities,
            "no records retrieved; a first-pass retrieval is required",
        )

    if state.missing_entities:
        return SufficiencyReport(
            SufficiencyVerdict.MISSING_SUBJECT,
            tuple(f"no record mentions {entity}" for entity in state.missing_entities),
            state.missing_entities,
            "the question's own subject is absent from the retrieved evidence",
        )

    # Chase a dangling link only when nothing already binds an asked-about
    # entity to a value; otherwise a candidate answer is already in hand and
    # the follow-up would spend a retrieval call for nothing.
    if state.bridge_entities and not state.answer_bearing_ids:
        return SufficiencyReport(
            SufficiencyVerdict.MISSING_BRIDGE,
            tuple(
                f"{entity} is referenced by the evidence but never resolved"
                for entity in state.bridge_entities
            ),
            state.bridge_entities,
            "evidence introduces a linking entity that no retrieved record resolves",
        )

    operation = detect_operation(state)
    wants_calculation = (
        require_calculation if require_calculation is not None
        else operation is not None and len(state.observed_operands) >= 2
    )
    if wants_calculation and operation is not None:
        return SufficiencyReport(
            SufficiencyVerdict.NEEDS_CALCULATION, (), (),
            "all operands are present and the evidence states a derivation rule",
            detected_operation=operation,
        )

    if state.contradictions:
        return SufficiencyReport(
            SufficiencyVerdict.CONFLICTING,
            tuple(
                f"{row['entity']} is bound to multiple values {row['values']}"
                for row in state.contradictions
            ),
            (),
            "retrieved records disagree; resolution requires temporal or source precedence",
        )

    return SufficiencyReport(
        SufficiencyVerdict.SUFFICIENT, (), (),
        "every question entity is covered and no unresolved link remains",
    )
