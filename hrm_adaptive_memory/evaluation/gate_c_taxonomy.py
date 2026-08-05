"""Gate C failure taxonomy.

An aggregate score does not say what to build next. Every failed iterative
retrieval is attributed to one stage of the pipeline, so the remaining gap can
be assigned to bridge detection, query formulation, retrieval, packing, or the
reader itself.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

SLOT_ECHO = re.compile(r"^\s*\[E\d+\]")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


class GateCFailure(str, Enum):
    NONE = "NONE"
    C1_BRIDGE_NOT_DETECTED = "C1_BRIDGE_NOT_DETECTED"
    C2_FALSE_BRIDGE = "C2_FALSE_BRIDGE"
    C3_AMBIGUOUS_BRIDGE = "C3_AMBIGUOUS_BRIDGE"
    C4_MALFORMED_QUERY = "C4_MALFORMED_QUERY"
    C5_RETRIEVAL_MISS = "C5_RETRIEVAL_MISS"
    C6_PACKER_DROPPED_REQUIRED = "C6_PACKER_DROPPED_REQUIRED"
    C7_DISTRACTOR_DERAILED_READER = "C7_DISTRACTOR_DERAILED_READER"
    C8_READER_REASONING = "C8_READER_REASONING"
    C9_CALCULATOR_ERROR = "C9_CALCULATOR_ERROR"
    C10_VERIFIER_ERROR = "C10_VERIFIER_ERROR"
    C11_CONTEXT_BUDGET_OVERFLOW = "C11_CONTEXT_BUDGET_OVERFLOW"
    C12_UNKNOWN = "C12_UNKNOWN"


@dataclass(frozen=True)
class GateCAttribution:
    task_id: str
    family: str
    arm: str
    failure: GateCFailure
    quality: float
    rationale: str
    missing_required_ids: tuple[str, ...]
    slot_label_echo: bool

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["failure"] = self.failure.value
        return row


def classify_gate_c(
    *, task_id: str, family: str, arm: str, quality: float, answer: str, output: str,
    required_ids: Iterable[str], first_pass_ids: Sequence[str],
    second_pass_ids: Sequence[str] = (), merged_ids: Sequence[str] = (),
    selected_ids: Sequence[str] = (), followup_query: str | None = None,
    bridge_entities: Sequence[str] = (), calculation: Mapping[str, Any] | None = None,
    budget_overflow: bool = False,
) -> GateCAttribution:
    required = set(required_ids)
    first = set(first_pass_ids)
    merged = set(merged_ids) or (first | set(second_pass_ids))
    selected = set(selected_ids) or merged
    missing_overall = tuple(sorted(required - selected))
    echo = bool(SLOT_ECHO.match(output))

    def result(failure: GateCFailure, rationale: str) -> GateCAttribution:
        return GateCAttribution(
            task_id, family, arm, failure, quality, rationale, missing_overall, echo,
        )

    if quality >= 1.0:
        return result(GateCFailure.NONE, "verified correct")

    if budget_overflow:
        return result(GateCFailure.C11_CONTEXT_BUDGET_OVERFLOW,
                      "evidence packet exceeded the working-context budget")

    # A correct answer stated but not in the verifier's scoring position.
    numbers = _NUMBER.findall(output)
    if answer in numbers and (not numbers or numbers[-1] != answer):
        return result(GateCFailure.C10_VERIFIER_ERROR,
                      "gold answer present in the output but outside the verifier's scoring position")

    if calculation is not None and not calculation.get("verified", False):
        return result(GateCFailure.C9_CALCULATOR_ERROR,
                      f"calculator refused or failed: {calculation.get('rationale', '')}")
    if calculation is not None and calculation.get("result") not in (None, "") \
            and calculation.get("result") != answer:
        return result(GateCFailure.C9_CALCULATOR_ERROR,
                      f"calculator returned {calculation.get('result')!r} for gold {answer!r}")

    still_missing = required - selected
    if still_missing:
        missing_after_first = required - first
        if not missing_after_first:
            # Everything needed was already in pass one, so the loss came later.
            if required <= merged:
                return result(GateCFailure.C6_PACKER_DROPPED_REQUIRED,
                              "required evidence survived retrieval but the packer dropped it")
            return result(GateCFailure.C12_UNKNOWN,
                          "required evidence present in pass one but absent after merge")
        if followup_query is None:
            if bridge_entities:
                return result(GateCFailure.C3_AMBIGUOUS_BRIDGE,
                              f"bridge candidates {list(bridge_entities)} found but no follow-up issued")
            return result(GateCFailure.C1_BRIDGE_NOT_DETECTED,
                          "evidence was incomplete and no bridge entity was detected")
        if len(bridge_entities) > 1:
            return result(GateCFailure.C3_AMBIGUOUS_BRIDGE,
                          f"multiple bridge candidates {list(bridge_entities)}; follow-up may have chased the wrong one")
        if not second_pass_ids:
            return result(GateCFailure.C4_MALFORMED_QUERY,
                          f"follow-up query {followup_query!r} returned nothing")
        return result(GateCFailure.C5_RETRIEVAL_MISS,
                      f"follow-up query {followup_query!r} ran but missed {sorted(still_missing)}")

    # Everything required is in the packet, so the failure is downstream of it.
    if followup_query is not None and not (required & set(second_pass_ids)) and len(bridge_entities) > 1:
        return result(GateCFailure.C2_FALSE_BRIDGE,
                      "follow-up chased an entity that contributed no required evidence")
    distractors = len(selected - required)
    if echo or distractors:
        return result(GateCFailure.C7_DISTRACTOR_DERAILED_READER,
                      f"complete evidence packed alongside {distractors} distractors"
                      + ("; reader emitted an evidence slot label" if echo else ""))
    if output.strip():
        return result(GateCFailure.C8_READER_REASONING,
                      "complete, clean evidence packed; the reader still answered wrongly")
    return result(GateCFailure.C12_UNKNOWN, "empty output with complete evidence")


def summarize_gate_c(rows: Sequence[GateCAttribution]) -> dict[str, Any]:
    counts = {value.value: 0 for value in GateCFailure}
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row.failure.value] += 1
        by_family.setdefault(row.family, {v.value: 0 for v in GateCFailure})[row.failure.value] += 1
    failures = [row for row in rows if row.failure != GateCFailure.NONE]
    stage = {
        "bridge_detection": sum(
            counts[value.value] for value in (
                GateCFailure.C1_BRIDGE_NOT_DETECTED, GateCFailure.C2_FALSE_BRIDGE,
                GateCFailure.C3_AMBIGUOUS_BRIDGE,
            )
        ),
        "query_formulation": counts[GateCFailure.C4_MALFORMED_QUERY.value],
        "retrieval": counts[GateCFailure.C5_RETRIEVAL_MISS.value],
        "packing": counts[GateCFailure.C6_PACKER_DROPPED_REQUIRED.value],
        "reader": sum(
            counts[value.value] for value in (
                GateCFailure.C7_DISTRACTOR_DERAILED_READER, GateCFailure.C8_READER_REASONING,
            )
        ),
        "tooling": sum(
            counts[value.value] for value in (
                GateCFailure.C9_CALCULATOR_ERROR, GateCFailure.C10_VERIFIER_ERROR,
                GateCFailure.C11_CONTEXT_BUDGET_OVERFLOW,
            )
        ),
        "unknown": counts[GateCFailure.C12_UNKNOWN.value],
    }
    return {
        "task_count": len(rows),
        "failure_count": len(failures),
        "counts": counts,
        "per_family": dict(sorted(by_family.items())),
        "by_stage": stage,
        "dominant_stage": max(stage, key=stage.get) if failures else None,
        "slot_label_echoes": sum(1 for row in rows if row.slot_label_echo),
    }
