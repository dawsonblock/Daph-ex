"""The oracle ladder must be independent of the mechanism it measures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from hrm_adaptive_memory.experiments.oracle_ladder import (
    LADDER_DELTAS,
    LadderArm,
    OracleMetadataMissing,
    decompose,
    oracle_bridge_query,
    read_oracle_facts,
)

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "data" / "hrm" / "controlled_gate_a_v4"


def tasks(split: str):
    return [json.loads(l) for l in
            (V4 / split / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]


def test_oracle_facts_come_from_metadata_not_from_the_extractor():
    """The v3 defect: the oracle re-ran extract_entities and died with it."""

    from hrm_adaptive_memory.evidence.state import extract_entities

    chain = [t for t in tasks("ood") if t["_oracle_metadata"]["latent_bridge"]]
    assert chain, "OOD must contain chain tasks with a latent bridge"
    task = chain[0]
    # The mechanism under test finds nothing on OOD surface forms...
    assert not extract_entities(task["question"])
    # ...yet the oracle still has the bridge.
    facts = read_oracle_facts(task)
    assert facts.has_bridge
    assert facts.bridge_surface
    assert facts.target_relation


def test_oracle_refuses_to_run_without_generator_truth():
    with pytest.raises(OracleMetadataMissing, match="must not fall back"):
        read_oracle_facts({"task_id": "x", "required_evidence_ids": []})


def test_r2_queries_the_bridge_and_r3_adds_the_target_relation():
    """R2 vs R3 is what separates bridge identification from query formulation."""

    task = [t for t in tasks("qualification") if t["_oracle_metadata"]["latent_bridge"]][0]
    facts = read_oracle_facts(task)
    r2 = oracle_bridge_query(facts, include_relation=False)
    r3 = oracle_bridge_query(facts, include_relation=True)
    assert r2 == facts.bridge_surface
    assert r3 == f"{facts.bridge_surface} {facts.target_relation}"
    assert r3 != r2, "R2 and R3 must issue different queries or the split is vacuous"


def test_non_chain_tasks_have_no_bridge_query():
    single = [t for t in tasks("qualification")
              if not t["_oracle_metadata"]["latent_bridge"]]
    assert single
    facts = read_oracle_facts(single[0])
    assert oracle_bridge_query(facts, include_relation=True) is None


def test_oracle_query_never_contains_the_answer():
    for split in ("qualification", "ood"):
        for task in tasks(split):
            facts = read_oracle_facts(task)
            for flag in (False, True):
                query = oracle_bridge_query(facts, include_relation=flag)
                if query is None:
                    continue
                assert task["answer"].lower() not in query.lower(), (
                    f"{task['task_id']}: oracle query leaks the answer")


def test_decomposition_names_every_step_and_isolates_reader_error():
    quality = {
        LadderArm.R0_ONE_PASS.value: 0.20,
        LadderArm.R1_CURRENT_TWO_PASS.value: 0.30,
        LadderArm.R2_ORACLE_BRIDGE_IDENTITY.value: 0.45,
        LadderArm.R3_ORACLE_BRIDGE_AND_RELATION.value: 0.55,
        LadderArm.R4_ORACLE_QUERY_ORACLE_SELECTION.value: 0.60,
        LadderArm.R5_ORACLE_EVIDENCE.value: 0.80,
    }
    out = decompose(quality)
    assert out["iteration"] == 0.10
    assert out["bridge_identification"] == 0.15
    assert out["query_formulation"] == 0.10
    assert out["selection"] == 0.05
    assert out["retrieval_ranking"] == 0.20
    assert out["reader_task_interface_error"] == 0.20
    assert out["_reader_error_is_not_retrieval_headroom"] is True
    # The retrieval terms must not silently absorb reader error.
    retrieval_total = sum(out[k] for k in LADDER_DELTAS)
    assert abs(retrieval_total - (0.80 - 0.20)) < 1e-9


def test_every_v4_task_can_drive_the_ladder():
    for split in ("development", "qualification", "ood"):
        for task in tasks(split):
            facts = read_oracle_facts(task)
            assert facts.target_relation
            assert facts.required_ids
