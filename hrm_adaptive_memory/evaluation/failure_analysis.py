"""Attribute every wrong answer to the stage that actually failed.

If all required evidence was retrieved and reached the prompt, a wrong answer
is not a retrieval failure. Conflating the two is what makes teams add more
retrieval infrastructure to fix a reasoning problem.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class FailureClass(str, Enum):
    NONE = "NONE"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    PACKING_FAILURE = "PACKING_FAILURE"
    REASONING_FAILURE = "REASONING_FAILURE"
    CALCULATION_FAILURE = "CALCULATION_FAILURE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    UNKNOWN = "UNKNOWN"


ARITHMETIC_FAMILIES = frozenset({"numeric_derivation"})


@dataclass(frozen=True)
class FailureAttribution:
    task_id: str
    family: str
    failure_class: FailureClass
    quality: float
    required_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    missing_required_ids: tuple[str, ...]
    dropped_in_packing_ids: tuple[str, ...]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["failure_class"] = self.failure_class.value
        return row


def _numbers(text: str) -> list[str]:
    return re.findall(r"[-+]?\d+(?:\.\d+)?", text)


def classify(
    *, task_id: str, family: str, quality: float, answer: str, output: str,
    required_ids: Iterable[str], retrieved_ids: Sequence[str],
    prompt_evidence_ids: Sequence[str] | None = None,
    evidence_contents: Mapping[str, str] | None = None,
) -> FailureAttribution:
    required = set(required_ids)
    retrieved = set(retrieved_ids)
    # Control arms rewrite evidence IDs as "<origin>#b1:<n>"; compare origins.
    packed = {
        str(value).split("#", 1)[0]
        for value in (retrieved_ids if prompt_evidence_ids is None else prompt_evidence_ids)
    }
    missing = tuple(sorted(required - retrieved))
    dropped = tuple(sorted((required & retrieved) - packed))

    if quality >= 1.0:
        return FailureAttribution(
            task_id, family, FailureClass.NONE, quality, tuple(sorted(required)),
            tuple(retrieved_ids), (), (), "verified correct",
        )

    # The verifier reads the last number; a correct answer stated earlier and
    # followed by another number is a verification artifact, not a model error.
    output_numbers = _numbers(output)
    if answer in output_numbers and (not output_numbers or output_numbers[-1] != answer):
        return FailureAttribution(
            task_id, family, FailureClass.VERIFICATION_FAILURE, quality,
            tuple(sorted(required)), tuple(retrieved_ids), missing, dropped,
            "gold answer present in output but not in the verifier's scoring position",
        )

    if missing:
        return FailureAttribution(
            task_id, family, FailureClass.RETRIEVAL_FAILURE, quality,
            tuple(sorted(required)), tuple(retrieved_ids), missing, dropped,
            f"{len(missing)} of {len(required)} required evidence records never retrieved",
        )

    if dropped:
        return FailureAttribution(
            task_id, family, FailureClass.PACKING_FAILURE, quality,
            tuple(sorted(required)), tuple(retrieved_ids), missing, dropped,
            f"{len(dropped)} required records retrieved but absent from the final prompt",
        )

    if family in ARITHMETIC_FAMILIES:
        # Every operand was present and packed, so a wrong number is arithmetic.
        operands_present = True
        if evidence_contents:
            packed_text = " ".join(
                evidence_contents.get(value, "") for value in required
            )
            operands_present = bool(_numbers(packed_text))
        if operands_present and output_numbers:
            return FailureAttribution(
                task_id, family, FailureClass.CALCULATION_FAILURE, quality,
                tuple(sorted(required)), tuple(retrieved_ids), missing, dropped,
                "all operands available and packed; produced number is arithmetically wrong",
            )

    if output.strip():
        return FailureAttribution(
            task_id, family, FailureClass.REASONING_FAILURE, quality,
            tuple(sorted(required)), tuple(retrieved_ids), missing, dropped,
            "complete evidence reached the prompt; answer is still wrong",
        )

    return FailureAttribution(
        task_id, family, FailureClass.UNKNOWN, quality, tuple(sorted(required)),
        tuple(retrieved_ids), missing, dropped, "empty output with complete evidence",
    )


def summarize(rows: Sequence[FailureAttribution]) -> dict[str, Any]:
    counts: dict[str, int] = {value.value: 0 for value in FailureClass}
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        counts[row.failure_class.value] += 1
        family = by_family.setdefault(row.family, {value.value: 0 for value in FailureClass})
        family[row.failure_class.value] += 1
    failures = [row for row in rows if row.failure_class != FailureClass.NONE]
    return {
        "task_count": len(rows),
        "failure_count": len(failures),
        "counts": counts,
        "per_family": dict(sorted(by_family.items())),
        "retrieval_bound_fraction": round(
            counts[FailureClass.RETRIEVAL_FAILURE.value] / max(1, len(failures)), 4
        ),
    }
