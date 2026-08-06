"""controlled_gate_a_v3 must be able to break the v2-saturating mechanism."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from hrm_adaptive_memory.experiments.generalization_dataset import (
    ITERATIVE_FAMILIES,
    OpportunityGroup,
    _ANSWER_BEARING_KINDS,
    _leaks,
    build_generalization_corpus,
)

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "data" / "hrm" / "controlled_gate_a_v3"
SPLITS = ("development", "qualification", "ood")


def load(split: str):
    directory = V3 / split
    tasks = [json.loads(l) for l in (directory / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]
    evidence = [json.loads(l) for l in (directory / "evidence.jsonl").read_text().splitlines() if l.strip()]
    return tasks, evidence


def test_generator_is_reproducible():
    first = build_generalization_corpus(seed=11, tasks_per_family=4, split="development")
    second = build_generalization_corpus(seed=11, tasks_per_family=4, split="development")
    assert first.manifest == second.manifest


@pytest.mark.parametrize("split", SPLITS)
def test_no_answer_leaks_into_question_or_distractor(split):
    tasks, evidence = load(split)
    by_id = {row["evidence_id"]: row for row in evidence}
    for task in tasks:
        assert not _leaks(task["question"], task["answer"]), task["task_id"]
        for row in evidence:
            if not row["evidence_id"].startswith(task["task_id"] + "/"):
                continue
            if row["metadata"]["record_kind"] in _ANSWER_BEARING_KINDS:
                continue
            assert not _leaks(row["content"], task["answer"]), (
                f"{task['task_id']}: answer leaked into distractor {row['evidence_id']}"
            )
        for required in task["required_evidence_ids"]:
            assert required in by_id, f"{task['task_id']} references missing evidence"


def test_iterative_structure_spans_many_families():
    """v2's fatal flaw: bridge structure in one family of five."""

    tasks, _ = load("qualification")
    families = {task["family"] for task in tasks}
    iterative = {task["family"] for task in tasks if task["metadata"]["iterative_family"]}
    assert len(families) >= 8, "a family bootstrap needs enough families to resample"
    assert len(iterative) >= 5, "iterative opportunity must not be confined to one family"


def test_retrieval_opportunity_is_heterogeneous():
    """Gate D is meaningless unless a follow-up can also be useless or harmful."""

    tasks, _ = load("qualification")
    groups = Counter(task["metadata"]["opportunity_group"] for task in tasks)
    assert set(groups) == {value.value for value in OpportunityGroup}
    for name, count in groups.items():
        assert count >= 25, f"{name} is too rare to measure ({count})"


def test_structural_variation_targets_are_met():
    tasks, _ = load("qualification")
    assert len({task["template_id"] for task in tasks}) >= 40
    assert len({task["source_cluster_id"] for task in tasks}) >= 20


def test_ood_split_holds_out_whole_styles_and_naming_regimes():
    qualification, _ = load("qualification")
    ood, _ = load("ood")
    q_styles = {t["metadata"]["source_style"] for t in qualification}
    o_styles = {t["metadata"]["source_style"] for t in ood}
    q_regimes = {t["metadata"]["entity_regime"] for t in qualification}
    o_regimes = {t["metadata"]["entity_regime"] for t in ood}
    assert not (q_styles & o_styles), "OOD shares a source style with qualification"
    assert not (q_regimes & o_regimes), "OOD shares an entity regime with qualification"


def test_answers_are_not_only_numeric():
    tasks, _ = load("qualification")
    kinds = Counter(task["metadata"]["answer_kind"] for task in tasks)
    assert len(kinds) >= 4, "results must not depend on numeric extraction"
    assert kinds["numeric"] < len(tasks) / 2


def test_exact_identifier_shortcut_is_broken():
    """Not every task may be solvable by exact token overlap."""

    tasks, _ = load("qualification")
    regimes = Counter(task["metadata"]["entity_regime"] for task in tasks)
    assert len(regimes) >= 2
    non_exact = sum(count for name, count in regimes.items() if name != "exact_id")
    assert non_exact > len(tasks) / 2, "most tasks still rely on exact identifiers"


def test_splits_were_frozen_before_evaluation():
    declared = json.loads((V3 / "SPLITS.json").read_text())
    assert declared["frozen_before_evaluation"] is True
    assert all(declared["structural_requirements"].values())
    for split in SPLITS:
        tasks, _ = load(split)
        manifest = json.loads((V3 / split / "dataset_manifest.json").read_text())
        assert manifest["task_count"] == len(tasks)


def test_style_holdout_is_checked_on_evidence_records_not_task_labels():
    """v3's holdout test read task metadata and passed while the claim was false.

    `task["metadata"]["source_style"]` records only the first record's style, so
    a chain whose second hop drew from the global style tuple could smuggle a
    held-out style into the split. Any corpus claiming a style holdout must be
    checked against every evidence record.
    """

    _, qualification_evidence = load("qualification")
    _, ood_evidence = load("ood")
    q_styles = {row["metadata"]["source_style"] for row in qualification_evidence}
    o_styles = {row["metadata"]["source_style"] for row in ood_evidence}
    overlap = q_styles & o_styles
    if V3.name == "controlled_gate_a_v3":
        # v3 is retained as historical evidence with this defect documented.
        limitations = V3 / "V3_KNOWN_LIMITATIONS.md"
        assert limitations.exists(), (
            "v3 violates its own style holdout and must ship the erratum documenting it"
        )
        assert overlap, "erratum describes a violation that is no longer present"
        return
    assert not overlap, f"evidence-level style holdout violated: {sorted(overlap)}"


def test_v3_limitations_are_recorded_rather_than_silently_fixed():
    text = (V3 / "V3_KNOWN_LIMITATIONS.md").read_text()
    for required in ("Source-style holdout is violated", "does not produce aliases",
                     "not answerable from evidence", "is not an oracle"):
        assert required in text, f"erratum omits: {required}"
