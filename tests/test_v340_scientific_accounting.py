import importlib.util
import inspect
import json
from pathlib import Path

import pytest
import torch

from daph.e3_metrics import (
    E3QualificationConfig,
    grouped_bootstrap,
    lambda_sweep,
    qualify_e3_pairs,
)
from daph.e3_protocol import (
    ClaimStrength,
    EvidenceMetadata,
    ExperimentScale,
    ExperimentTier,
    profile_stability,
    promote_e3_placement,
    write_evidence_metadata,
)
from daph.effort_frontier import build_effort_frontier, qualify_oracle_opportunity
from daph.hard_case import E3HardCaseMiner, HardCaseMiningConfig, HardCaseRecord
from daph.verified_tasks import (
    calibrated_sensitivity_split,
    choose_calibration_families,
    generate_verified_tasks,
    natural_heldout_split,
)
from scripts.run_e3_hardcase_ablation import _load_profile_selection


def pair(
    task_id, *, q2=0.0, q3=1.0, c2=1.0, c3=1.1,
    e2=False, e3=True, template=None, family="math",
):
    return {
        "task_id": task_id, "quality_e2": q2, "quality_e3": q3,
        "compute_e2": c2, "compute_e3": c3,
        "e2_correct": e2, "e3_correct": e3,
        "task_family": family, "template_id": template or task_id,
        "difficulty": "MEDIUM",
    }


def test_utility_requires_actual_compute_not_effort_id_inference():
    row = pair("a")
    del row["compute_e3"]
    row["effort_mode"] = "fixed_3"
    with pytest.raises(ValueError, match="receipt-backed"):
        qualify_e3_pairs([row])


def test_default_smoke_gate_cannot_pass_with_two_tasks():
    report = qualify_e3_pairs([pair("a"), pair("b")], E3QualificationConfig(bootstrap_samples=50))
    assert report["qualification_status"] == "INSUFFICIENT_POWER"
    assert not report["qualified"]


def test_qualification_tier_cannot_be_promoted_by_low_n_even_with_local_override():
    scale = ExperimentScale(
        tier=ExperimentTier.QUALIFICATION, heldout_examples=500,
        training_seeds=(1, 2, 3), evaluation_seed=4,
    )
    rows = [pair("a"), pair("b")]
    report = qualify_e3_pairs(rows, E3QualificationConfig(
        min_tasks=2, bootstrap_samples=50, experiment_scale=scale,
    ))
    assert report["qualification_status"] == "INSUFFICIENT_POWER"
    assert "OBSERVED_HELDOUT_BELOW_TIER_MINIMUM" in report["experiment_scale"]["failures"]


def test_repeated_seed_rows_do_not_inflate_unique_qualification_task_count():
    scale = ExperimentScale(
        tier=ExperimentTier.QUALIFICATION, heldout_examples=500,
        training_seeds=(1, 2, 3), evaluation_seed=4,
    )
    rows = []
    for seed in (1, 2, 3):
        for task_id in ("a", "b"):
            row = pair(task_id, template=f"template_{task_id}")
            row["training_seed"] = seed
            rows.append(row)
    report = qualify_e3_pairs(rows, E3QualificationConfig(
        min_tasks=2, bootstrap_samples=50, experiment_scale=scale,
    ))
    assert report["unique_task_count"] == 2
    assert report["qualification_status"] == "INSUFFICIENT_POWER"


def test_qualification_scale_rejects_a_declared_low_n_run():
    scale = ExperimentScale(
        tier=ExperimentTier.QUALIFICATION, heldout_examples=24,
        training_seeds=(1,), evaluation_seed=1,
    )
    with pytest.raises(ValueError, match="QUALIFICATION experiment scale"):
        scale.validate()


def test_predeclared_qualification_pass_requires_observed_tasks_groups_and_seeds():
    scale = ExperimentScale(
        tier=ExperimentTier.QUALIFICATION, heldout_examples=500,
        training_seeds=(11, 22, 33), evaluation_seed=44,
    )
    rows = []
    for index in range(500):
        row = pair(f"q{index}", template=f"template_{index % 9}")
        row["training_seed"] = (11, 22, 33)[index % 3]
        rows.append(row)
    report = qualify_e3_pairs(rows, E3QualificationConfig(
        bootstrap_samples=50, experiment_scale=scale,
    ))
    assert report["experiment_scale"]["passed"]
    assert report["qualification_status"] == "PASS_QUALITY_AND_UTILITY"


def test_quality_and_cost_aware_gates_are_distinct():
    rows = [pair(f"t{i}", c3=3.0) for i in range(24)]
    report = qualify_e3_pairs(rows, E3QualificationConfig(bootstrap_samples=200, seed=1))
    assert report["quality_gate"]["passed"]
    assert not report["utility_gate"]["passed"]
    assert report["qualification_status"] == "PASS_QUALITY_FAIL_UTILITY"


def test_lambda_changes_utility_decision_without_changing_quality():
    rows = [pair(f"t{i}", c3=1.75) for i in range(24)]
    free = qualify_e3_pairs(rows, E3QualificationConfig(lambda_compute=0, bootstrap_samples=100))
    expensive = qualify_e3_pairs(rows, E3QualificationConfig(lambda_compute=2, bootstrap_samples=100))
    assert free["quality_lcb95"] == expensive["quality_lcb95"]
    assert free["utility_gate"]["passed"]
    assert not expensive["utility_gate"]["passed"]


def test_break_even_lambda_is_delta_quality_over_delta_compute():
    report = lambda_sweep(
        [pair("a", q3=0.5, c3=1.25), pair("b", q3=0.5, c3=1.25)],
        [0, 1, 2], bootstrap_samples=50,
    )
    assert report["aggregate_break_even_lambda"] == pytest.approx(2.0)
    assert report["results"][-1]["mean_delta_utility"] == pytest.approx(0.0)


def test_quality_only_statistic_is_never_named_verified_utility():
    report = qualify_e3_pairs([pair(f"t{i}") for i in range(4)], E3QualificationConfig(bootstrap_samples=50))
    assert "mean_verified_utility_delta" not in report
    assert "mean_quality_delta" in report and "mean_utility_delta" in report


def test_grouped_bootstrap_is_deterministic_and_resamples_whole_templates():
    rows = [
        {"template_id": "a", "delta": 1.0}, {"template_id": "a", "delta": 1.0},
        {"template_id": "b", "delta": -1.0}, {"template_id": "b", "delta": -1.0},
    ]
    first = grouped_bootstrap(rows, "delta", group_key="template_id", samples=200, confidence=0.95, seed=7)
    second = grouped_bootstrap(rows, "delta", group_key="template_id", samples=200, confidence=0.95, seed=7)
    assert first == second and first["group_count"] == 2


def test_calibrated_and_natural_splits_have_distinct_provenance():
    tasks = generate_verified_tasks(count_per_family=2, seed=3)
    natural, natural_manifest = natural_heldout_split(tasks, count=6, seed=4)
    outcomes = [{"task_id": row["task_id"], "e2_correct": index % 2 == 0} for index, row in enumerate(tasks)]
    calibrated, calibrated_manifest = calibrated_sensitivity_split(tasks, outcomes, count=6, seed=4)
    assert natural_manifest["split_type"] == "NATURAL_HELDOUT"
    assert calibrated_manifest["split_type"] == "CALIBRATED_SENSITIVITY"
    assert not natural_manifest["e2_outcomes_inspected"] and calibrated_manifest["e2_outcomes_inspected"]
    assert len(natural) == len(calibrated) == 6


def test_calibrated_split_is_family_stratified_not_globally_class_sampled():
    tasks = generate_verified_tasks(count_per_family=4, seed=30)
    outcomes = []
    for task in tasks:
        index = int(task["task_id"].rsplit("-", 1)[1])
        outcomes.append({"task_id": task["task_id"], "e2_correct": index % 2 == 0})
    selected, manifest = calibrated_sensitivity_split(tasks, outcomes, count=18, seed=3)
    assert manifest["family_stratified"]
    assert {item["selected_tasks"] for item in manifest["per_task_family"].values()} == {2}
    assert sum(item["selected_e2_successes"] for item in manifest["per_task_family"].values()) == 9
    assert len(selected) == 18


def test_calibration_family_fallback_excludes_infeasible_family_without_e3_outcomes():
    tasks = generate_verified_tasks(count_per_family=12, seed=31)
    outcomes = []
    for task in tasks:
        index = int(task["task_id"].rsplit("-", 1)[1])
        correct = index % 2 == 0
        if task["task_family"] == "code_output":
            correct = False
        outcomes.append({"task_id": task["task_id"], "e2_correct": correct})
    selected, manifest = choose_calibration_families(
        tasks, outcomes, split_counts=(18,), minimum_families=5,
    )
    assert len(selected) == 8
    assert "code_output" in manifest["excluded_families"]
    assert not manifest["e3_outcomes_inspected"]


def test_natural_test_selection_does_not_inspect_e3_outcomes():
    tasks = generate_verified_tasks(count_per_family=2, seed=8)
    poisoned = [{**row, "e3_correct": index % 2 == 0} for index, row in enumerate(tasks)]
    opposite = [{**row, "e3_correct": index % 2 != 0} for index, row in enumerate(tasks)]
    left, _ = natural_heldout_split(poisoned, count=8, seed=9)
    right, _ = natural_heldout_split(opposite, count=8, seed=9)
    assert [row["task_id"] for row in left] == [row["task_id"] for row in right]


def _hard_record(task_id, category, correct):
    return HardCaseRecord(
        task_id=task_id, category=category, e2_correct=correct,
        e2_verifier_reward=float(correct), e2_entropy=None, e2_confidence=None,
        e2_ce=None, e2_answer=None, task_family="math", difficulty="MEDIUM",
        task_digest=task_id, task_payload={"task_id": task_id},
    )


def test_hard_case_miner_reports_and_respects_configured_mix():
    miner = E3HardCaseMiner(
        torch.nn.Linear(1, 1), lambda *_: {"correct": True},
        HardCaseMiningConfig(hard_failure_ratio=0.5, hard_uncertain_ratio=0.25, easy_correct_ratio=0.25),
    )
    records = [
        _hard_record("f", "HARD_FAILURE", False),
        _hard_record("u", "HARD_UNCERTAIN", True),
        _hard_record("e", "EASY_CORRECT", True),
    ]
    _, manifest = miner.sample_with_manifest(records, 8)
    assert manifest["realized_counts"] == {"HARD_FAILURE": 4, "HARD_UNCERTAIN": 2, "EASY_CORRECT": 2}
    assert manifest["e2_successes"] and manifest["e2_failures"]


def test_generated_tasks_have_required_verification_metadata():
    rows = generate_verified_tasks(count_per_family=1, seed=1)
    assert len({row["task_family"] for row in rows}) >= 5
    assert all(row["template_id"] and row["difficulty"] and row["generator_version"] and row["verifier_version"] for row in rows)


def test_generator_emits_multiple_templates_and_labels_difficulty_source_honestly():
    rows = generate_verified_tasks(count_per_family=6, seed=1)
    additions = {row["template_id"] for row in rows if row["task_family"] == "addition_with_carry"}
    assert len(additions) == 3
    assert all(row["difficulty"].startswith("GENERATOR_") for row in rows)
    assert all(row["difficulty_source"] == "generator_numeric_scale_v1" for row in rows)


def test_profile_stability_metrics_detect_stable_and_unstable_rankings():
    stable = profile_stability({1: {0: 0.1, 1: 0.9, 2: 0.5}, 2: {0: 0.2, 1: 0.8, 2: 0.6}}, top_k=2)
    unstable = profile_stability({1: {0: 0.1, 1: 0.9, 2: 0.5}, 2: {0: 0.9, 1: 0.1, 2: 0.5}}, top_k=1)
    assert stable["mean_spearman"] == pytest.approx(1.0)
    assert stable["stable_for_promotion"]
    assert not unstable["stable_for_promotion"]


def test_profile_stability_ranks_only_the_shared_layer_subset():
    report = profile_stability({
        1: {0: 0.0, 1: 10.0, 2: 20.0, 10: 5.0, 11: 6.0},
        2: {0: 0.0, 1: 10.0, 2: 20.0},
    }, top_k=2)
    assert report["mean_spearman"] == pytest.approx(1.0)


def test_heuristic_profiled_promotion_is_deterministic_and_stability_gated():
    candidates = [
        {"name": "HEURISTIC_MIDDLE", "quality_lcb95": 0.02, "utility_lcb95": 0.01, "rescues": 5, "regressions": 1, "seed_pass_rate": 1.0, "compute_delta": 0.04},
        {"name": "PROFILED_LAYER", "quality_lcb95": 0.03, "utility_lcb95": 0.02, "rescues": 6, "regressions": 1, "seed_pass_rate": 1.0, "compute_delta": 0.04},
    ]
    assert promote_e3_placement(candidates, profile_stable=False)["canonical"] == "HEURISTIC_MIDDLE"
    assert promote_e3_placement(
        candidates, profile_stable=True, profile_tier_passed=True,
        experiment_scale_passed=True, natural_test_passed=True,
    )["canonical"] == "PROFILED_LAYER"


def test_profiled_placement_cannot_promote_without_passing_profile_tier():
    candidate = [{"name": "PROFILED_LAYER", "quality_lcb95": 0.02, "utility_lcb95": 0.01, "rescues": 3, "regressions": 0, "seed_pass_rate": 1.0, "compute_delta": 0.04}]
    report = promote_e3_placement(
        candidate, profile_stable=True, profile_tier_passed=False,
        experiment_scale_passed=True, natural_test_passed=True,
    )
    assert not report["promoted"]


def test_aggregated_profile_is_accepted_only_with_bound_promotion_evidence(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "profile_status": "AGGREGATED_PROFILE", "profile_digest": "aggregate-digest",
    }))
    (tmp_path / "rankings.json").write_text(json.dumps({"best_contiguous_region": [11, 12, 13]}))
    (tmp_path / "profile_tier_validation.json").write_text(json.dumps({
        "tier": "PROFILE_PILOT", "passed": True, "promotion_passed": True,
        "profile_stability": {"stable_for_promotion": True},
    }))
    layers, digest, status, tier = _load_profile_selection(tmp_path)
    assert layers == [11, 12, 13]
    assert digest == "aggregate-digest" and status == "AGGREGATED_PROFILE"
    assert tier["promotion_passed"]


def frontier_rows(e1_quality=0.8, e1_compute=0.8):
    return [{
        "task_id": f"t{i}", "task_family": "math", "template_id": f"g{i}",
        "efforts": {
            "E0": {"quality": 0.4, "compute": 0.5},
            "E1": {"quality": e1_quality, "compute": e1_compute},
            "E2": {"quality": 0.9, "compute": 1.0},
            "E3": {"quality": 1.0 if i % 2 else 0.9, "compute": 1.1},
        },
    } for i in range(6)]


def test_effort_frontier_marks_dominated_arms():
    report = build_effort_frontier(frontier_rows(e1_quality=0.3, e1_compute=0.8), lambdas=[0, 1])
    assert "E1" in report["dominated_arms"]
    assert report["arms"]["E2"]["frontier_status"] == "ANCHOR"


def test_oracle_uses_per_task_actual_compute():
    cheap = qualify_oracle_opportunity(
        frontier_rows(), lambda_compute=0.1, qualified_non_e2_arms=["E3"], bootstrap_samples=100,
    )
    expensive_rows = frontier_rows()
    for row in expensive_rows:
        row["efforts"]["E3"]["compute"] = 100.0
    expensive = qualify_oracle_opportunity(
        expensive_rows, lambda_compute=0.1, qualified_non_e2_arms=["E3"], bootstrap_samples=100,
    )
    assert cheap["mean_fixed_utility"]["E3"] > expensive["mean_fixed_utility"]["E3"]


def test_policy_training_remains_blocked_if_oracle_or_arm_gate_fails():
    no_arm = qualify_oracle_opportunity(
        frontier_rows(), lambda_compute=1.0, qualified_non_e2_arms=[], bootstrap_samples=50,
    )
    assert not no_arm["policy_training_allowed"]
    assert no_arm["reason"] == "NO_QUALIFIED_NON_E2_ARM"


def metadata():
    return EvidenceMetadata(
        artifact_commit="abc", repository_version="3.4.0", test_count_at_creation=157,
        pytest_digest="p", config_digest="c", source_tree_digest="s",
        claim_strength=ClaimStrength.MECHANISM_SIGNAL,
    )


def test_evidence_artifact_stores_commit_and_repository_version(tmp_path):
    path = write_evidence_metadata(tmp_path, metadata())
    values = json.loads(path.read_text())
    assert values["artifact_commit"] == "abc" and values["repository_version"] == "3.4.0"


def test_historical_artifact_metadata_is_not_silently_overwritten(tmp_path):
    write_evidence_metadata(tmp_path, metadata())
    with pytest.raises(FileExistsError):
        write_evidence_metadata(tmp_path, metadata())


def test_real_model_script_imports_without_transformers_installed():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("e3_hardcase_import_test", root / "scripts" / "run_e3_hardcase_ablation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    source = inspect.getsource(module.main)
    assert "from transformers import" in source
