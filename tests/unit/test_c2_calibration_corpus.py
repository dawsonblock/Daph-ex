"""controlled_gate_c2_calibration_v1: separation from V4 and regime balance."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from hrm_adaptive_memory.experiments.c2_calibration_dataset import (
    ANSWER_BEARING, HEADS, ROLES, SOURCE_STYLES, leaks)
from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (
    _HEADS as V4_HEADS, _ROLES as V4_ROLES, SOURCE_STYLES as V4_STYLES, verify_inferable)

ROOT = Path(__file__).resolve().parents[2]
CAL = ROOT / "data" / "hrm" / "controlled_gate_c2_calibration_v1"
PARTS = ("c2_cal_id", "c2_cal_surface", "c2_cal_holdout")


def load(part):
    d = CAL / part
    t = [json.loads(l) for l in (d / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    e = [json.loads(l) for l in (d / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return t, e


def test_vocabulary_is_disjoint_from_v4():
    """A calibration corpus sharing V4's vocabulary would not be a separate set."""
    assert not (set(HEADS) & set(V4_HEADS))
    assert not (set(ROLES) & set(V4_ROLES))
    assert not (set(SOURCE_STYLES) & set(V4_STYLES))


def test_evidence_ids_and_task_ids_cannot_collide_with_v4():
    v4 = set()
    for split in ("development", "qualification", "ood"):
        p = ROOT / "data/hrm/controlled_gate_a_v4" / split / "evidence.jsonl"
        v4 |= {json.loads(l)["evidence_id"] for l in p.read_text().splitlines() if l.strip()}
    for part in PARTS:
        tasks, evidence = load(part)
        assert not ({r["evidence_id"] for r in evidence} & v4)
        assert all(t["task_id"].startswith("c2cal-") for t in tasks)


def test_v4_is_untouched_by_this_corpus():
    audit = json.loads((ROOT / "data/hrm/controlled_gate_a_v4" / "AUDIT.json").read_text())
    assert audit["audit"]["VALID_V4"] is True


@pytest.mark.parametrize("part", PARTS)
def test_regimes_are_balanced(part):
    tasks, _ = load(part)
    counts = Counter(t["metadata"]["entity_regime"] for t in tasks)
    assert len(set(counts.values())) == 1, f"unbalanced: {counts}"


def test_surface_partition_carries_the_regimes_v4_development_lacked():
    """The reason this corpus exists."""
    tasks, _ = load("c2_cal_surface")
    assert {t["metadata"]["entity_regime"] for t in tasks} == {"alias", "description"}


def test_id_partition_carries_bm25_favourable_regimes():
    tasks, _ = load("c2_cal_id")
    assert {t["metadata"]["entity_regime"] for t in tasks} == {"canonical", "abbreviation"}


def test_holdout_covers_all_four_regimes_and_is_marked_reserved():
    tasks, _ = load("c2_cal_holdout")
    assert {t["metadata"]["entity_regime"] for t in tasks} == {
        "canonical", "abbreviation", "alias", "description"}
    # Structured state, not prose. Three prior failures came from asserting
    # exact strings against narrative text I had written separately; tests
    # should validate state and let prose be rendered from it.
    state = json.loads((CAL / "AUDIT.json").read_text())["state"]
    assert state["holdout_partition"] == "c2_cal_holdout"
    assert state["holdout_status"] == "RESERVED"
    assert state["holdout_runs_permitted"] == 1


@pytest.mark.parametrize("part", PARTS)
def test_every_task_is_inferable_and_leak_free(part):
    tasks, evidence = load(part)
    by_id = {r["evidence_id"]: r for r in evidence}
    for task in tasks:
        assert not verify_inferable(task, by_id), task["task_id"]
        assert not leaks(task["question"], task["answer"])
        for row in evidence:
            if row["evidence_id"].startswith(task["task_id"] + "/") and \
                    row["metadata"]["record_kind"] not in ANSWER_BEARING:
                assert not leaks(row["content"], task["answer"])


def test_valid_gate_recorded():
    audit = json.loads((CAL / "AUDIT.json").read_text())
    assert audit["VALID_C2_CAL"] is True
    assert not audit["problems"]
    state = audit["state"]
    assert state["replaces_v4"] is False
    assert state["purpose"] == "gate_c2_component_selection"
    assert state["frozen_before_evaluation"] is True
    assert state["valid"] is True


def test_promotion_protocol_is_frozen_before_any_run():
    """The reference arm and every threshold must be fixed in advance."""

    protocol = json.loads((ROOT / "configs" / "gate_c2_protocol.json").read_text())
    assert protocol["frozen_before_any_calibration_run"] is True
    assert protocol["reference_arm"]["immutable"] is True
    assert protocol["reference_arm"]["name"] == "P0_bm25"
    assert protocol["primary_metric"] == "complete_set@50"
    rules = protocol["per_regime_rules"]
    assert rules["canonical"]["threshold"] == -0.02
    assert rules["abbreviation"]["threshold"] == -0.02
    assert rules["description"]["threshold"] == 0.10
    # Alias must never be collapsed into an aggregate.
    assert rules["alias"]["direction"] == "report_only"
    assert protocol["aggregate_verdict_forbidden"] is True


def test_selection_gate_threshold_is_predeclared():
    protocol = json.loads((ROOT / "configs" / "gate_c2_protocol.json").read_text())
    gate = protocol["selection_gate"]
    assert isinstance(gate["tau_selection"], (int, float))
    assert gate["tau_selection"] > 0
    assert gate["conditional_ceiling_required"]


def test_holdout_may_not_inform_any_choice():
    protocol = json.loads((ROOT / "configs" / "gate_c2_protocol.json").read_text())
    forbidden = set(protocol["holdout_policy"]["may_never_inform"])
    for kind in ("model choice", "hyperparameter choice", "query-template choice",
                 "fusion-weight choice", "packet-budget choice", "selector choice",
                 "threshold calibration", "debugging"):
        assert kind in forbidden
    assert protocol["holdout_policy"]["runs_permitted"] == 1


def test_regime_aware_policy_is_blocked_until_alias_mechanism_exists():
    protocol = json.loads((ROOT / "configs" / "gate_c2_protocol.json").read_text())
    assert "BLOCKED" in protocol["policy_arms"]["P3_regime_aware_deterministic"]
    assert "never a production arm" in protocol["policy_arms"]["P4_oracle_regime_policy"]


def test_alias_decomposition_metrics_are_required():
    protocol = json.loads((ROOT / "configs" / "gate_c2_protocol.json").read_text())
    required = set(protocol["alias_decomposition_required"])
    assert {"identity_record_recall_at_k", "canonical_entity_recovered",
            "target_relation_record_recall_at_k"} <= required
