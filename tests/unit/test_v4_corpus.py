"""Adversarial tests for controlled_gate_a_v4, one per v3 defect.

Each test inspects the artifact it makes a claim about. v3's holdout test read
`task["metadata"]["source_style"]` — a label — and passed while the corpus
violated the property in both directions. These read evidence records, rendered
text, and proof graphs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from hrm_adaptive_memory.experiments.generalization_dataset_v4 import (
    EntityRegime,
    OpportunityGroup,
    _ANSWER_BEARING_KINDS,
    _leaks,
    build_v4_corpus,
    verify_inferable,
)

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "data" / "hrm" / "controlled_gate_a_v4"
SPLITS = ("development", "qualification", "ood")


def load(split: str):
    directory = V4 / split
    tasks = [json.loads(l) for l in (directory / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (directory / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence


def norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


# ---- DEFECT 1: style holdout must hold over EVERY evidence record ----------

def test_style_holdout_holds_over_every_evidence_record():
    _, q_evidence = load("qualification")
    _, o_evidence = load("ood")
    q = {row["metadata"]["source_style"] for row in q_evidence}
    o = {row["metadata"]["source_style"] for row in o_evidence}
    assert not (q & o), f"evidence-level style overlap: {sorted(q & o)}"
    # And no record may carry a style outside its split's declared set.
    for split in SPLITS:
        _, evidence = load(split)
        allowed = set(json.loads(
            (V4 / split / "dataset_manifest.json").read_text())["allowed_source_styles"])
        actual = {row["metadata"]["source_style"] for row in evidence}
        assert actual <= allowed, f"{split} records escaped: {sorted(actual - allowed)}"


def test_second_hop_and_temporal_records_respect_the_split():
    """The exact v3 bug: chain second hops and temporal records reaching past the split."""

    for split in SPLITS:
        _, evidence = load(split)
        allowed = set(json.loads(
            (V4 / split / "dataset_manifest.json").read_text())["allowed_source_styles"])
        for kind in ("required", "required_current", "superseded", "dead_end_link",
                     "rejected_candidate", "direct_answer", "required_identity"):
            styles = {row["metadata"]["source_style"] for row in evidence
                      if row["metadata"]["record_kind"] == kind}
            assert styles <= allowed, f"{split}/{kind} escaped: {sorted(styles - allowed)}"


# ---- DEFECT 2: aliases must be genuinely different names -------------------

def test_alias_is_not_a_truncation_of_the_canonical_name():
    """v3 turned 'Nimbus assembly' into the query 'Nimbus As'."""

    tasks, evidence = load("ood")
    by_id = {row["evidence_id"]: row for row in evidence}
    alias_tasks = [t for t in tasks if t["metadata"]["entity_regime"] == "alias"]
    assert alias_tasks, "OOD must contain alias tasks"
    for task in alias_tasks:
        surfaces = task["_oracle_metadata"]["surfaces"]
        subject, canonical = surfaces["subject"], surfaces["canonical"]
        assert not canonical.startswith(subject), (
            f"{task['task_id']}: alias {subject!r} is a prefix of {canonical!r}")
        assert not subject.startswith(canonical), (
            f"{task['task_id']}: alias {subject!r} merely extends {canonical!r}")
        # A real alias shares no head word with the canonical name.
        assert subject.split()[0] != canonical.split()[0], (
            f"{task['task_id']}: alias reuses the canonical head word")


def test_every_non_canonical_reference_has_explicit_identity_evidence():
    for split in SPLITS:
        tasks, evidence = load(split)
        by_id = {row["evidence_id"]: row for row in evidence}
        for task in tasks:
            if task["metadata"]["entity_regime"] == "canonical":
                continue
            identity = [v for v in task["required_evidence_ids"] if v.endswith("/identity")]
            assert identity, f"{task['task_id']} references a non-canonical name with no identity record"
            content = norm(by_id[identity[0]]["content"])
            surfaces = task["_oracle_metadata"]["surfaces"]
            assert norm(surfaces["subject"]) in content
            assert norm(surfaces["canonical"]) in content


# ---- DEFECT 3: every task must be answerable from visible evidence ---------

@pytest.mark.parametrize("split", SPLITS)
def test_every_task_is_inferable_from_visible_evidence(split):
    """v3 shipped 120 OOD tasks whose subject appeared nowhere in their evidence."""

    tasks, evidence = load(split)
    by_id = {row["evidence_id"]: row for row in evidence}
    failures = {t["task_id"]: verify_inferable(t, by_id) for t in tasks}
    failures = {k: v for k, v in failures.items() if v}
    assert not failures, f"{len(failures)} non-inferable tasks: {list(failures)[:3]}"


@pytest.mark.parametrize("split", SPLITS)
def test_question_subject_appears_in_the_required_evidence(split):
    tasks, evidence = load(split)
    by_id = {row["evidence_id"]: row for row in evidence}
    for task in tasks:
        text = " || ".join(norm(by_id[v]["content"])
                           for v in task["required_evidence_ids"] if v in by_id)
        subject = norm(task["_oracle_metadata"]["surfaces"]["subject"])
        assert subject in text, f"{task['task_id']}: subject {subject!r} unfindable"


# ---- DEFECT 4: the oracle must be independent of the extractor under test --

def test_proof_graph_exists_and_names_latent_identities():
    tasks, _ = load("qualification")
    for task in tasks:
        meta = task["_oracle_metadata"]
        assert meta["proof_edges"], f"{task['task_id']} has no proof graph"
        assert meta["latent_subject"]
        assert meta["target_relation"]
        assert meta["answer_node"]


def test_latent_identifiers_never_reach_runtime_visible_text():
    pattern = re.compile(r"#(subject|bridge|value|decoy|near)\b|entity_\d+|value_\d+")
    for split in SPLITS:
        tasks, evidence = load(split)
        for row in evidence:
            assert not pattern.search(row["content"]), (
                f"{row['evidence_id']} leaks a latent identifier")
        for task in tasks:
            assert not pattern.search(task["question"])
            assert not pattern.search(task["answer"])


def test_oracle_bridge_is_available_without_running_the_extractor():
    """The ladder must read identity from metadata, not re-derive it from text."""

    tasks, _ = load("ood")
    chain_tasks = [t for t in tasks if t["metadata"]["iterative_family"]]
    assert chain_tasks
    with_bridge = [t for t in chain_tasks if t["_oracle_metadata"]["latent_bridge"]]
    assert with_bridge, "no OOD chain task exposes a latent bridge"
    from hrm_adaptive_memory.evidence.state import extract_entities
    # The extractor under test finds nothing here; the oracle must not depend on it.
    assert not extract_entities(with_bridge[0]["question"])
    assert with_bridge[0]["_oracle_metadata"]["latent_bridge"]


# ---- leakage and structure -------------------------------------------------

@pytest.mark.parametrize("split", SPLITS)
def test_no_answer_leaks(split):
    tasks, evidence = load(split)
    prefix: dict[str, list] = {}
    for row in evidence:
        prefix.setdefault(row["evidence_id"].split("/", 1)[0], []).append(row)
    for task in tasks:
        assert not _leaks(task["question"], task["answer"]), task["task_id"]
        for row in prefix.get(task["task_id"], []):
            if row["metadata"]["record_kind"] in _ANSWER_BEARING_KINDS:
                continue
            assert not _leaks(row["content"], task["answer"]), (
                f"{task['task_id']}: leak into {row['evidence_id']}")


def test_structural_diversity_targets():
    tasks, _ = load("qualification")
    assert len({t["family"] for t in tasks}) >= 8
    assert len({t["family"] for t in tasks if t["metadata"]["iterative_family"]}) >= 5
    assert len({t["template_id"] for t in tasks}) >= 40
    assert len({t["source_cluster_id"] for t in tasks}) >= 20
    groups = Counter(t["metadata"]["opportunity_group"] for t in tasks)
    assert set(groups) == {v.value for v in OpportunityGroup}
    assert len({t["metadata"]["answer_kind"] for t in tasks}) >= 4


def test_valid_v4_gate_recorded_and_true():
    audit = json.loads((V4 / "AUDIT.json").read_text())
    assert audit["audit"]["VALID_V4"] is True
    assert audit["frozen_before_evaluation"] is True
    assert not audit["audit"]["problems"]


def test_generator_refuses_to_emit_a_non_inferable_corpus():
    """The build must fail closed, not warn."""

    import hrm_adaptive_memory.experiments.generalization_dataset_v4 as v4

    original = v4.verify_inferable
    v4.verify_inferable = lambda task, by_id: ["synthetic failure"]
    try:
        with pytest.raises(RuntimeError, match="not inferable"):
            v4.build_v4_corpus(seed=1, tasks_per_family=1, split="development",
                               styles=["formal_registry"], regimes=["canonical"])
    finally:
        v4.verify_inferable = original
