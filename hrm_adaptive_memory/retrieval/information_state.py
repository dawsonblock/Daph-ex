"""Information state that ACCUMULATES across retrieval hops.

The measured defect this exists to fix: the old follow-up did

    subject -> find bridge -> discard subject -> search bridge

and the development sweep showed what that costs. Complete-set@50 by follow-up
formulation:

    bridge alone                  0.500
    bridge + relation             0.625
    subject + bridge + relation   0.900

Retaining the subject is worth +0.275 over bridge+relation and +0.400 over the
bridge alone -- larger than any encoder change measured so far. The earlier
Q0-Q3 ladder showed the same thing from the other side: querying the bridge
alone scored *worse* than the original question and collapsed to 0.000 on OOD.

The architectural principle, which is stronger than a query-string change:

    Retrieval iteration must ACCUMULATE information state, never replace
    earlier context with the latest bridge.

Regression tests pin that invariant, and the bridge-only formulation is kept as
a named negative control so the defect cannot silently return.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class InformationState:
    """What is known so far. Frozen, so a hop must produce a new state."""

    subject: str
    target_relation: str
    bridge: str | None = None
    resolved_identities: tuple[tuple[str, str], ...] = ()
    hop: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject.strip():
            raise ValueError("InformationState requires a subject; losing it is the defect")
        if not self.target_relation.strip():
            raise ValueError("InformationState requires a target relation")

    def with_bridge(self, bridge: str, *, record_id: str | None = None) -> "InformationState":
        """Add a bridge WITHOUT dropping the subject or relation."""

        return replace(
            self, bridge=bridge, hop=self.hop + 1,
            provenance={**dict(self.provenance), f"hop{self.hop + 1}_bridge_record": record_id},
        )

    def with_identity(self, surface: str, canonical: str, *,
                      record_id: str | None = None) -> "InformationState":
        """Record an alias/description -> canonical resolution, additively.

        Alias regimes need this extra transition:
            alias -> identity record -> canonical -> update state -> retrieve
        """

        return replace(
            self,
            resolved_identities=self.resolved_identities + ((surface, canonical),),
            hop=self.hop + 1,
            provenance={**dict(self.provenance), f"hop{self.hop + 1}_identity_record": record_id},
        )

    @property
    def canonical_subject(self) -> str:
        """The subject under its most recent resolution, original as fallback."""

        for surface, canonical in reversed(self.resolved_identities):
            if surface == self.subject:
                return canonical
        return self.subject


# Frozen on calibration; the winner of the development formulation sweep.
FOLLOWUP_FORMULATION = "subject_bridge_relation"


def formulate_followup(state: InformationState, *,
                       formulation: str = FOLLOWUP_FORMULATION) -> str:
    """Render a follow-up query from accumulated state.

    `bridge_only` is retained as a NEGATIVE CONTROL: it reproduces the defect
    (0.500 vs 0.900) and exists so the regression is measurable, not so it can
    be selected.
    """

    subject = state.canonical_subject
    bridge = state.bridge
    if formulation == "subject_bridge_relation":
        parts = [subject] + ([bridge] if bridge else []) + [state.target_relation]
    elif formulation == "bridge_relation":
        parts = ([bridge] if bridge else [subject]) + [state.target_relation]
    elif formulation == "bridge_only":  # negative control
        parts = [bridge or subject]
    else:
        raise ValueError(f"Unknown follow-up formulation {formulation!r}")
    return " ".join(part for part in parts if part)
