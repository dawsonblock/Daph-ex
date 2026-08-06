"""Independent R0–R5 oracle ladder for Gate C1 on controlled_gate_a_v4.

The v3 `oracle_bridge` arm derived its "oracle" bridge by re-running
`extract_entities` — the very component under test — so it failed identically
wherever extraction failed and could not decompose anything. Every oracle arm
here reads latent identity from the task's evaluator-only proof graph instead.

The ladder separates bridge *identification* from query *formulation*, which
the v3 instrument conflated:

    R0  one-pass                     current retrieval + current selector
    R1  current two-pass             current bridge heuristic + query + selector
    R2  oracle bridge identity       true bridge surface, CURRENT query style
                                     R2 − R1 = bridge identification headroom
    R3  oracle bridge + relation     true bridge AND true target relation
                                     R3 − R2 = query formulation headroom
    R4  oracle query + oracle select  R3's query, but perfect selection
                                     R4 − R3 = selection headroom
    R5  oracle evidence              the required set, handed over directly
                                     R5 − R4 = residual retrieval/ranking
                                     1 − R5  = reader/task/interface error

`1 − R5` is error relative to perfect task accuracy under the current model and
prompt. It is not headroom belonging to any retrieval component and must never
be summed with the terms above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class LadderArm(str, Enum):
    R0_ONE_PASS = "R0_one_pass"
    R1_CURRENT_TWO_PASS = "R1_current_two_pass"
    R2_ORACLE_BRIDGE_IDENTITY = "R2_oracle_bridge_identity"
    R3_ORACLE_BRIDGE_AND_RELATION = "R3_oracle_bridge_and_relation"
    R4_ORACLE_QUERY_ORACLE_SELECTION = "R4_oracle_query_oracle_selection"
    R5_ORACLE_EVIDENCE = "R5_oracle_evidence"


LADDER_ORDER = tuple(LadderArm)

# What each successive difference measures. Kept next to the arms so a report
# cannot silently relabel a raw arm difference as a causal attribution.
LADDER_DELTAS = {
    "iteration": (LadderArm.R0_ONE_PASS, LadderArm.R1_CURRENT_TWO_PASS),
    "bridge_identification": (LadderArm.R1_CURRENT_TWO_PASS,
                              LadderArm.R2_ORACLE_BRIDGE_IDENTITY),
    "query_formulation": (LadderArm.R2_ORACLE_BRIDGE_IDENTITY,
                          LadderArm.R3_ORACLE_BRIDGE_AND_RELATION),
    "selection": (LadderArm.R3_ORACLE_BRIDGE_AND_RELATION,
                  LadderArm.R4_ORACLE_QUERY_ORACLE_SELECTION),
    "retrieval_ranking": (LadderArm.R4_ORACLE_QUERY_ORACLE_SELECTION,
                          LadderArm.R5_ORACLE_EVIDENCE),
}


class OracleMetadataMissing(RuntimeError):
    """Raised when an oracle arm is asked to run without generator truth."""


@dataclass(frozen=True)
class OracleFacts:
    """Evaluator-only truth for one task, read from the proof graph."""

    latent_subject: str
    latent_bridge: str | None
    target_relation: str
    bridge_surface: str | None
    subject_surface: str
    required_ids: tuple[str, ...]

    @property
    def has_bridge(self) -> bool:
        return self.latent_bridge is not None and bool(self.bridge_surface)


def read_oracle_facts(task: Mapping[str, Any]) -> OracleFacts:
    """Extract generator truth. Never calls the mechanism under test."""

    meta = task.get("_oracle_metadata")
    if not meta:
        raise OracleMetadataMissing(
            f"task {task.get('task_id')!r} has no _oracle_metadata; the oracle ladder "
            "must not fall back to deriving identity from surface text"
        )
    surfaces = meta.get("surfaces", {})
    bridge_surface = surfaces.get("bridge")
    return OracleFacts(
        latent_subject=meta["latent_subject"],
        latent_bridge=meta.get("latent_bridge"),
        target_relation=meta["target_relation"],
        bridge_surface=bridge_surface,
        subject_surface=surfaces.get("subject", ""),
        required_ids=tuple(task["required_evidence_ids"]),
    )


def oracle_bridge_query(facts: OracleFacts, *, include_relation: bool) -> str | None:
    """R2 queries the bridge alone; R3 adds the target relation.

    R2 mirrors what the current mechanism does once it has the right entity:
    search the entity's name. R3 asks for the missing *relation on* that
    entity, which is what an information-gap formulation would produce.
    """

    if not facts.has_bridge:
        return None
    if include_relation:
        return f"{facts.bridge_surface} {facts.target_relation}"
    return facts.bridge_surface


@dataclass
class LadderReceipt:
    arm: str
    task_id: str
    selected_ids: tuple[str, ...]
    oracle_query: str | None
    used_oracle_metadata: bool
    retrieval_calls: int
    complete_set_success: float
    extra: dict[str, Any] = field(default_factory=dict)


def decompose(quality_by_arm: Mapping[str, float]) -> dict[str, Any]:
    """Named differences plus the reader term, kept explicitly separate."""

    out: dict[str, Any] = {}
    for name, (lower, upper) in LADDER_DELTAS.items():
        low, high = quality_by_arm.get(lower.value), quality_by_arm.get(upper.value)
        if low is not None and high is not None:
            out[name] = round(high - low, 4)
    ceiling = quality_by_arm.get(LadderArm.R5_ORACLE_EVIDENCE.value)
    if ceiling is not None:
        out["reader_task_interface_error"] = round(1.0 - ceiling, 4)
        out["_reader_error_is_not_retrieval_headroom"] = True
    return out
