"""Contracts for the paired selector analysis.

The point of the paired analysis is to distinguish "few tasks changed a lot"
from "many tasks changed in both directions" when the means happen to coincide.
These tests construct both shapes with an identical mean and assert the analysis
separates them, so a future refactor cannot quietly collapse back to comparing
means.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "analyze_selector_ladder", ROOT / "scripts/analyze_selector_ladder.py")
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def _row(task_id, arm, quality, *, dropped=(), cluster="c0", template="t0", family="f0",
         csr_eligible=True, answer=True):
    return {
        "partition": "P", "budget": 6, "arm": arm, "task_id": task_id,
        "family": family, "template_id": template, "source_cluster_id": cluster,
        "quality": quality, "csr_eligible": csr_eligible, "csr_ok": quality == 1.0,
        "role_retained": {"answer": answer, "bridge": "bridge" not in dropped,
                          "identity": "identity" not in dropped},
        "roles_available": {"answer": ["a"], "bridge": ["b"], "identity": ["c"]},
        "roles_dropped": list(dropped),
    }


def _pair(base_specs, arm_specs):
    base = {t: _row(t, "S0_raw", q, cluster=c, family=f)
            for t, q, c, f in base_specs}
    arm = {t: _row(t, "ARM", q, dropped=d, cluster=c, family=f)
           for t, q, d, c, f in arm_specs}
    return arm, base


def test_paired_delta_separates_churn_from_quiet_change():
    """Two arms with the SAME mean delta but very different task-level shape."""
    n = 20
    base_specs = [(f"t{i}", 1.0 if i % 2 else 0.0, f"c{i % 4}", f"f{i % 5}") for i in range(n)]

    # Churny: 6 improved, 4 harmed -> net +2/20 = +0.10
    churn = []
    for i, (t, q, c, f) in enumerate(base_specs):
        if i < 6 and q == 0.0:
            churn.append((t, 1.0, (), c, f))
        elif 6 <= i < 10 and q == 1.0:
            churn.append((t, 0.0, (), c, f))
        else:
            churn.append((t, q, (), c, f))
    arm, base = _pair(base_specs, churn)
    churn_result = analysis.paired_delta(arm, base, "quality")

    # Quiet: only improvements, no regressions at all.
    quiet = [(t, 1.0 if (i < 4 and q == 0.0) else q, (), c, f)
             for i, (t, q, c, f) in enumerate(base_specs)]
    arm2, base2 = _pair(base_specs, quiet)
    quiet_result = analysis.paired_delta(arm2, base2, "quality")

    assert churn_result["negative_tasks"] > 0
    assert quiet_result["negative_tasks"] == 0
    # The shapes differ even though both are net-positive arms.
    assert churn_result["positive_tasks"] != quiet_result["positive_tasks"]


def test_paired_delta_reports_zero_for_an_identical_arm():
    base_specs = [(f"t{i}", float(i % 2), f"c{i % 3}", f"f{i % 3}") for i in range(12)]
    identical = [(t, q, (), c, f) for t, q, c, f in base_specs]
    arm, base = _pair(base_specs, identical)
    result = analysis.paired_delta(arm, base, "quality")
    assert result["mean_delta"] == 0.0
    assert result["positive_tasks"] == result["negative_tasks"] == 0
    assert result["neutral_tasks"] == 12
    assert result["ci95"] == [0.0, 0.0]
    assert result["excludes_zero"] is False


def test_paired_delta_pairs_only_tasks_eligible_under_both_arms():
    base = {"t0": _row("t0", "S0_raw", 1.0, csr_eligible=True),
            "t1": _row("t1", "S0_raw", 1.0, csr_eligible=False)}
    arm = {"t0": _row("t0", "ARM", 0.0, csr_eligible=True),
           "t1": _row("t1", "ARM", 0.0, csr_eligible=True)}
    result = analysis.paired_delta(arm, base, "csr_ok",
                                  eligible=lambda r: r["csr_eligible"])
    assert result["paired_tasks"] == 1, "t1 is ineligible under S0 and must be excluded"


def test_paired_delta_ignores_tasks_absent_from_the_baseline():
    base = {"t0": _row("t0", "S0_raw", 1.0)}
    arm = {"t0": _row("t0", "ARM", 1.0), "ghost": _row("ghost", "ARM", 0.0)}
    assert analysis.paired_delta(arm, base, "quality")["paired_tasks"] == 1


def test_discard_analysis_attributes_regressions_to_the_role_s0_kept():
    base = {f"t{i}": _row(f"t{i}", "S0_raw", 1.0) for i in range(4)}
    arm = {
        "t0": _row("t0", "ARM", 0.0, dropped=("bridge",)),
        "t1": _row("t1", "ARM", 0.0, dropped=("bridge",)),
        "t2": _row("t2", "ARM", 0.0, dropped=("identity",)),
        "t3": _row("t3", "ARM", 1.0),  # no regression
    }
    result = analysis.discard_analysis(arm, base)
    assert result["regressions_vs_s0"] == 3
    assert result["role_lost_that_s0_kept"] == {"bridge": 2, "identity": 1}
    assert result["regressions_with_no_role_loss"] == 0


def test_discard_analysis_does_not_blame_a_role_s0_also_dropped():
    """If S0 dropped the bridge too, the bridge cannot explain the regression."""
    base = {"t0": _row("t0", "S0_raw", 1.0, dropped=("bridge",))}
    arm = {"t0": _row("t0", "ARM", 0.0, dropped=("bridge",))}
    result = analysis.discard_analysis(arm, base)
    assert result["regressions_vs_s0"] == 1
    assert result["role_lost_that_s0_kept"] == {}
    assert result["regressions_with_no_role_loss"] == 1


def test_grouped_bootstrap_resamples_groups_not_tasks():
    """A single group cannot produce spread, however many tasks it holds."""
    one_group = {"g0": [0.5] * 50}
    low, high = analysis.grouped_bootstrap_ci(one_group, resamples=200)
    assert low == high == 0.5

    # Groups that disagree must produce a non-degenerate interval.
    split = {"g0": [0.0] * 10, "g1": [1.0] * 10}
    low2, high2 = analysis.grouped_bootstrap_ci(split, resamples=2000)
    assert low2 < high2


def test_grouped_bootstrap_is_deterministic_under_a_fixed_seed():
    values = {f"g{i}": [float(i % 3)] * 5 for i in range(8)}
    first = analysis.grouped_bootstrap_ci(values, resamples=500, seed=7)
    second = analysis.grouped_bootstrap_ci(values, resamples=500, seed=7)
    assert first == second
