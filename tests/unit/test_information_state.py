"""Subject retention is a measured invariant, not a style preference."""

from __future__ import annotations

import pytest

from hrm_adaptive_memory.retrieval.information_state import (
    FOLLOWUP_FORMULATION, InformationState, formulate_followup)


def state():
    return InformationState(subject="Jacana pressure assembly",
                            target_relation="ownership tier")


def test_followup_preserves_original_subject():
    """The defect: subject -> find bridge -> DISCARD subject -> search bridge."""

    advanced = state().with_bridge("Sparrow relay unit")
    query = formulate_followup(advanced)
    assert "Jacana pressure assembly" in query, "subject was dropped between hops"
    assert "Sparrow relay unit" in query


def test_followup_preserves_target_relation():
    query = formulate_followup(state().with_bridge("Sparrow relay unit"))
    assert "ownership tier" in query


def test_bridge_augments_state_instead_of_replacing_it():
    before = state()
    after = before.with_bridge("Sparrow relay unit")
    assert after.subject == before.subject
    assert after.target_relation == before.target_relation
    assert after.bridge == "Sparrow relay unit"
    assert after.hop == before.hop + 1
    # Frozen dataclass: a hop cannot mutate the earlier state in place.
    assert before.bridge is None


def test_state_cannot_exist_without_a_subject_or_relation():
    with pytest.raises(ValueError, match="losing it is the defect"):
        InformationState(subject="  ", target_relation="tier")
    with pytest.raises(ValueError, match="target relation"):
        InformationState(subject="thing", target_relation="")


def test_identity_resolution_is_additive_and_updates_the_subject():
    """Alias path: alias -> identity record -> canonical -> retrieve."""

    s = InformationState(subject="Bluebird unit", target_relation="service region")
    resolved = s.with_identity("Bluebird unit", "Nimbus control module", record_id="e1")
    assert resolved.canonical_subject == "Nimbus control module"
    assert resolved.subject == "Bluebird unit", "the original surface must survive"
    assert resolved.resolved_identities == (("Bluebird unit", "Nimbus control module"),)
    query = formulate_followup(resolved.with_bridge("Kestrel relay unit"))
    assert "Nimbus control module" in query and "Kestrel relay unit" in query


def test_bridge_only_negative_control_reproduces_the_defect():
    advanced = state().with_bridge("Sparrow relay unit")
    control = formulate_followup(advanced, formulation="bridge_only")
    assert control == "Sparrow relay unit"
    assert "Jacana" not in control, "the negative control must actually drop the subject"
    assert formulate_followup(advanced) != control


def test_frozen_formulation_is_the_measured_winner():
    assert FOLLOWUP_FORMULATION == "subject_bridge_relation"


def test_provenance_accumulates_per_hop():
    s = state().with_identity("a", "b", record_id="id1").with_bridge("c", record_id="id2")
    assert s.provenance["hop1_identity_record"] == "id1"
    assert s.provenance["hop2_bridge_record"] == "id2"
    assert s.hop == 2
