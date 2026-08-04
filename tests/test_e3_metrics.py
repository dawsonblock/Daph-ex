from daph.e3_metrics import e3_pair_metrics


def test_e3_pair_metrics_counts_rescues_regressions_and_breakdowns():
    report = e3_pair_metrics([
        {"e2_correct": False, "e3_correct": True, "difficulty_bucket": "hard", "task_family": "arithmetic"},
        {"e2_correct": True, "e3_correct": False, "difficulty_bucket": "hard", "task_family": "arithmetic"},
        {"e2_correct": True, "e3_correct": True, "difficulty_bucket": "medium", "task_family": "logic"},
        {"e2_correct": False, "e3_correct": False, "difficulty_bucket": "medium", "task_family": "logic"},
    ])
    assert report["rescues"] == 1
    assert report["regressions"] == 1
    assert report["net_rescue_rate"] == 0.0
    assert report["by_difficulty"]["hard"]["tasks"] == 2
    assert report["by_task_family"]["arithmetic"]["e3_accuracy"] == 0.5
