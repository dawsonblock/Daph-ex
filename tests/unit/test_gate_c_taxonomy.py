"""Gate C taxonomy: every failure is attributed to the stage that caused it."""

from __future__ import annotations

import pytest

from hrm_adaptive_memory.evaluation.gate_c_taxonomy import (
    GateCFailure,
    classify_gate_c,
    summarize_gate_c,
)

BASE = dict(task_id="t", family="two_hop", arm="two_pass")


@pytest.mark.parametrize("name,kwargs,expected", [
    ("verified correct", dict(
        quality=1.0, answer="840", output="840", required_ids=["a", "b"],
        first_pass_ids=["a"], second_pass_ids=["b"], merged_ids=["a", "b"],
        selected_ids=["a", "b"], followup_query="Adapter-1"),
     GateCFailure.NONE),
    ("bridge never detected", dict(
        quality=0.0, answer="840", output="1", required_ids=["a", "b"],
        first_pass_ids=["a"], selected_ids=["a"], followup_query=None, bridge_entities=[]),
     GateCFailure.C1_BRIDGE_NOT_DETECTED),
    ("ambiguous bridge", dict(
        quality=0.0, answer="840", output="1", required_ids=["a", "b"],
        first_pass_ids=["a"], second_pass_ids=["z"], merged_ids=["a", "z"],
        selected_ids=["a", "z"], followup_query="X",
        bridge_entities=["Adapter-1", "Adapter-2"]),
     GateCFailure.C3_AMBIGUOUS_BRIDGE),
    ("query returned nothing", dict(
        quality=0.0, answer="840", output="1", required_ids=["a", "b"],
        first_pass_ids=["a"], second_pass_ids=[], merged_ids=["a"], selected_ids=["a"],
        followup_query="Adapter-1", bridge_entities=["Adapter-1"]),
     GateCFailure.C4_MALFORMED_QUERY),
    ("follow-up ran but missed", dict(
        quality=0.0, answer="840", output="1", required_ids=["a", "b"],
        first_pass_ids=["a"], second_pass_ids=["z"], merged_ids=["a", "z"],
        selected_ids=["a", "z"], followup_query="Adapter-1", bridge_entities=["Adapter-1"]),
     GateCFailure.C5_RETRIEVAL_MISS),
    ("packer dropped required", dict(
        quality=0.0, answer="840", output="1", required_ids=["a", "b"],
        first_pass_ids=["a", "b"], merged_ids=["a", "b"], selected_ids=["a"]),
     GateCFailure.C6_PACKER_DROPPED_REQUIRED),
    ("distractors derailed reader", dict(
        quality=0.0, answer="840", output="[E4]", required_ids=["a", "b"],
        first_pass_ids=["a", "b"], merged_ids=["a", "b", "x"], selected_ids=["a", "b", "x"]),
     GateCFailure.C7_DISTRACTOR_DERAILED_READER),
    ("clean evidence, wrong answer", dict(
        quality=0.0, answer="840", output="176", required_ids=["a", "b"],
        first_pass_ids=["a", "b"], merged_ids=["a", "b"], selected_ids=["a", "b"]),
     GateCFailure.C8_READER_REASONING),
    ("calculator wrong", dict(
        quality=0.0, answer="224", output="12", required_ids=["a"],
        first_pass_ids=["a"], selected_ids=["a"],
        calculation={"verified": True, "result": "12"}),
     GateCFailure.C9_CALCULATOR_ERROR),
    ("verifier scoring artefact", dict(
        quality=0.0, answer="840", output="840 as recorded in registry 12",
        required_ids=["a"], first_pass_ids=["a"], selected_ids=["a"]),
     GateCFailure.C10_VERIFIER_ERROR),
    ("budget overflow", dict(
        quality=0.0, answer="840", output="", required_ids=["a"],
        first_pass_ids=["a"], selected_ids=["a"], budget_overflow=True),
     GateCFailure.C11_CONTEXT_BUDGET_OVERFLOW),
])
def test_each_failure_mode_is_attributed_to_its_stage(name, kwargs, expected):
    assert classify_gate_c(**BASE, **kwargs).failure == expected, name


def test_retrieval_failure_is_not_blamed_on_the_reader():
    """The distinction that decides what to build next."""

    retrieval = classify_gate_c(**BASE, quality=0.0, answer="840", output="1",
                                required_ids=["a", "b"], first_pass_ids=["a"],
                                second_pass_ids=["z"], merged_ids=["a", "z"],
                                selected_ids=["a", "z"], followup_query="Adapter-1",
                                bridge_entities=["Adapter-1"])
    reader = classify_gate_c(**BASE, quality=0.0, answer="840", output="176",
                             required_ids=["a", "b"], first_pass_ids=["a", "b"],
                             merged_ids=["a", "b"], selected_ids=["a", "b"])
    assert retrieval.failure != reader.failure
    assert retrieval.missing_required_ids == ("b",)
    assert reader.missing_required_ids == ()


def test_summary_reports_dominant_stage_and_echo_count():
    rows = [
        classify_gate_c(**BASE, quality=0.0, answer="8", output="[E3]",
                        required_ids=["a"], first_pass_ids=["a"], selected_ids=["a", "x"]),
        classify_gate_c(**BASE, quality=0.0, answer="8", output="[E5]",
                        required_ids=["a"], first_pass_ids=["a"], selected_ids=["a", "x"]),
        classify_gate_c(**BASE, quality=1.0, answer="8", output="8",
                        required_ids=["a"], first_pass_ids=["a"], selected_ids=["a"]),
    ]
    summary = summarize_gate_c(rows)
    assert summary["failure_count"] == 2
    assert summary["dominant_stage"] == "reader"
    assert summary["slot_label_echoes"] == 2
    assert summary["counts"]["NONE"] == 1
