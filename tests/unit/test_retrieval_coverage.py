"""Gate C2 coverage metrics, and regression tests for two real fusion bugs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hrm_adaptive_memory.evaluation.retrieval_coverage import (
    RetrievalGroundTruth, score_coverage, summarize_coverage)

ROOT = Path(__file__).resolve().parents[2]
V4 = ROOT / "data" / "hrm" / "controlled_gate_a_v4"


def truths(split="qualification"):
    return [RetrievalGroundTruth.from_task(json.loads(l))
            for l in (V4 / split / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]


# ---- the two union bugs, locked in ----------------------------------------

def _capped_union(a, b, k):
    """BUG 1: capping the pool at k lets the first ranking fill it entirely."""
    return list(dict.fromkeys(a + b))[:k]


def _concatenated_union(a, b, limit):
    """BUG 2: concatenating puts every `a` result ahead of any `b` result."""
    return list(dict.fromkeys(a + b))[:limit]


def _interleaved_union(a, b, limit):
    merged, seen = [], set()
    for position in range(max(len(a), len(b))):
        for ranking in (a, b):
            if position < len(ranking) and ranking[position] not in seen:
                seen.add(ranking[position])
                merged.append(ranking[position])
                if len(merged) >= limit:
                    return merged
    return merged


def test_capped_union_silently_degrades_to_the_first_retriever():
    a = [f"a{i}" for i in range(50)]
    b = [f"b{i}" for i in range(50)]
    assert _capped_union(a, b, 50) == a, "regression fixture no longer reproduces the bug"
    assert _interleaved_union(a, b, 50) != a, "interleaving must expose the second retriever"


def test_concatenated_union_hides_the_second_retriever_at_every_measured_depth():
    a = [f"a{i}" for i in range(50)]
    b = [f"b{i}" for i in range(50)]
    concatenated = _concatenated_union(a, b, 100)
    for depth in (1, 5, 10, 20, 50):
        assert set(concatenated[:depth]) <= set(a), "b appears earlier than the bug allows"
    interleaved = _interleaved_union(a, b, 100)
    assert set(interleaved[:10]) & set(b), "interleaving must surface b within depth 10"


def test_interleaved_union_contains_both_rankings_and_deduplicates():
    a, b = ["x", "y", "z"], ["y", "w", "v"]
    merged = _interleaved_union(a, b, 10)
    assert merged == list(dict.fromkeys(merged))
    assert set(merged) == {"x", "y", "z", "w", "v"}


# ---- metrics ---------------------------------------------------------------

def test_complete_set_requires_every_record_but_partial_proof_does_not():
    truth = next(t for t in truths() if len(t.required_ids) >= 2)
    partial = score_coverage(truth, [truth.required_ids[0]], retriever="t")
    assert partial.complete_set_at[10] == 0.0
    assert 0.0 < partial.partial_proof_coverage_at[10] < 1.0
    full = score_coverage(truth, list(truth.required_ids), retriever="t")
    assert full.complete_set_at[10] == 1.0
    assert full.partial_proof_coverage_at[10] == pytest.approx(1.0)


def test_answer_bearing_records_carry_more_weight_than_ordinary_ones():
    """R4 answered 46% of tasks while holding complete evidence for 17%."""

    truth = next(t for t in truths()
                 if t.answer_record_ids and len(t.required_ids) >= 2
                 and set(t.answer_record_ids) < set(t.required_ids))
    weights = truth.weights()
    answer_weight = max(weights[v] for v in truth.answer_record_ids)
    other = [w for v, w in weights.items() if v not in truth.answer_record_ids]
    assert other and answer_weight > max(other)


def test_ground_truth_references_real_evidence_ids_not_latent_ones():
    """Proof labels must point at retrievable records, not generator internals."""

    evidence_ids = {json.loads(l)["evidence_id"] for l in
                    (V4 / "qualification" / "evidence.jsonl").read_text().splitlines() if l.strip()}
    for truth in truths()[:80]:
        for value in truth.required_ids + truth.proof_path_ids:
            assert value in evidence_ids, f"{value} is not a real evidence record"
            assert "#" not in value, f"{value} looks like a latent node id"


def test_summary_reports_every_axis():
    rows = truths()[:40]
    results = [score_coverage(t, list(t.required_ids), retriever="oracle") for t in rows]
    summary = summarize_coverage(results, {t.task_id: t for t in rows}, retriever="oracle")
    assert summary["overall"]["complete_set@50"] == 1.0
    for axis in ("family", "entity_regime", "answer_kind", "source_style", "opportunity_group"):
        assert summary["by_axis"][axis]


def cal_truths(part="c2_cal_surface"):
    root = ROOT / "data" / "hrm" / "controlled_gate_c2_calibration_v1" / part
    return [RetrievalGroundTruth.from_task(json.loads(l))
            for l in (root / "oracle_tasks.jsonl").read_text().splitlines() if l.strip()]


def test_identity_records_are_identified_for_alias_and_description_tasks():
    """Alias questions must traverse an identity record before the target."""

    truths = cal_truths()
    assert truths
    with_identity = [t for t in truths if t.identity_record_ids]
    assert with_identity, "surface partition must have identity records"
    for truth in with_identity[:20]:
        assert all(v.endswith("/identity") for v in truth.identity_record_ids)


def test_alias_decomposition_separates_identity_from_target():
    truth = next(t for t in cal_truths() if t.identity_record_ids and t.answer_record_ids)
    identity_only = score_coverage(truth, list(truth.identity_record_ids), retriever="t")
    assert identity_only.identity_record_found == 1.0
    assert identity_only.target_relation_record_found == 0.0
    target_only = score_coverage(truth, list(truth.answer_record_ids), retriever="t")
    assert target_only.identity_record_found == 0.0
    assert target_only.target_relation_record_found == 1.0


def test_identity_metrics_are_none_for_canonical_tasks():
    """Canonical tasks have no identity hop and must not dilute the statistic."""

    canonical = [t for t in cal_truths("c2_cal_id") if not t.identity_record_ids]
    assert canonical
    row = score_coverage(canonical[0], list(canonical[0].required_ids), retriever="t")
    assert row.identity_record_found is None


def test_summary_reports_conditional_target_recall():
    truths = cal_truths()
    rows = [score_coverage(t, list(t.required_ids), retriever="oracle") for t in truths]
    summary = summarize_coverage(rows, {t.task_id: t for t in truths}, retriever="oracle")
    o = summary["overall"]
    assert o["identity_record_recall_among_identity_tasks"] == 1.0
    assert o["target_recall_given_identity_found"] == 1.0
    assert o["identity_task_count"] == len([t for t in truths if t.identity_record_ids])
