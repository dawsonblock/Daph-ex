"""Unified E3 variant, location, and dose-response experiment contracts."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class E3ExperimentVariant:
    name: str
    refinement_mode: str
    location_fraction: float
    refinement_steps: int
    reuse_pretrained_layers: bool = False
    train_middle_layers: bool = False
    hard_case_curriculum: bool = True
    strong_e2_distillation: bool = False


@dataclass(frozen=True)
class E2DifficultyBandConfig:
    target_size: int
    min_accuracy: float = 0.30
    max_accuracy: float = 0.70
    target_accuracy: float = 0.50
    seed: int = 42

    def validate(self) -> None:
        if self.target_size < 2:
            raise ValueError("Difficulty-band target_size must be at least two")
        if not 0.0 <= self.min_accuracy <= self.target_accuracy <= self.max_accuracy <= 1.0:
            raise ValueError("Difficulty-band accuracies must satisfy 0 <= min <= target <= max <= 1")


def canonical_variant_matrix() -> List[E3ExperimentVariant]:
    return [
        E3ExperimentVariant("V0_E2", "none", 0.50, 0),
        E3ExperimentVariant("V1_FINAL", "final_refine", 1.00, 1),
        E3ExperimentVariant("V2_MIDDLE", "middle_recurrent", 0.50, 2),
        E3ExperimentVariant("V3_REPEAT", "middle_repeat", 0.50, 1, True),
        E3ExperimentVariant("V4_MIDDLE_TRAINABLE", "middle_recurrent", 0.50, 2, False, True),
        E3ExperimentVariant("V5_PROFILED", "profiled_middle_recurrent", 0.50, 2),
    ]


def dose_response_variants(steps: Sequence[int] = (0, 1, 2, 4, 8)) -> List[E3ExperimentVariant]:
    if any(step < 0 for step in steps):
        raise ValueError("Refinement step counts must be non-negative")
    return [E3ExperimentVariant(f"E3_{step}", "none" if step == 0 else "middle_recurrent", 0.50, step) for step in steps]


def location_ablation_variants(steps: int = 2) -> List[E3ExperimentVariant]:
    return [
        E3ExperimentVariant("EARLY", "middle_recurrent", 0.25, steps),
        E3ExperimentVariant("MIDDLE", "middle_recurrent", 0.50, steps),
        E3ExperimentVariant("LATE", "middle_recurrent", 0.75, steps),
        E3ExperimentVariant("FINAL", "final_refine", 1.00, steps),
    ]


def active_refinement_layer(model: Any) -> int:
    """Return the physical block containing the active E3 refiner."""
    if not hasattr(model, "e3_config") or not hasattr(model, "e3_region"):
        raise ValueError("E3 refinement location requires e3_config and e3_region")
    if not hasattr(model, "layers") or not model.layers:
        raise ValueError("E3 refinement location requires a non-empty layer stack")
    return (
        len(model.layers) - 1
        if model.e3_config.e3_refinement_mode == "final_refine"
        else int(model.e3_region.insertion_layer)
    )


def set_refinement_steps(model: Any, steps: int) -> None:
    """Set the canonical serialized and legacy E3 dose fields together."""
    value = int(steps)
    maximum = int(getattr(model.e3_config, "e3_max_refine_steps", value))
    if value < 1 or value > maximum:
        raise ValueError(f"E3 refinement steps must be within [1, {maximum}]")
    model.e3_config.e3_refine_steps = value
    model.default_e3_steps = value


def numeric_answer_correct(text: str, expected: Any) -> bool:
    """Compare the last generated number with the expected numeric answer."""
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not values:
        return False
    try:
        return float(values[-1]) == float(expected)
    except (TypeError, ValueError):
        return text.strip() == str(expected).strip()


def select_mixed_success_tasks(
    tasks: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    config: E2DifficultyBandConfig,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select a deterministic E2 mixed-success band without task duplication."""
    config.validate()
    if len(tasks) < config.target_size:
        raise ValueError(f"Need {config.target_size} candidates, received {len(tasks)}")
    by_id = {str(row["task_id"]): bool(row["e2_correct"]) for row in outcomes}
    if len(by_id) != len(outcomes):
        raise ValueError("Calibration outcomes contain duplicate task IDs")
    successes, failures = [], []
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id not in by_id:
            raise ValueError(f"Missing E2 calibration outcome for task {task_id}")
        (successes if by_id[task_id] else failures).append(dict(task))

    n = config.target_size
    minimum_successes = math.ceil(config.min_accuracy * n)
    maximum_successes = math.floor(config.max_accuracy * n)
    lower = max(minimum_successes, n - len(failures))
    upper = min(maximum_successes, len(successes))
    if lower > upper:
        raise ValueError(
            "Candidate pool cannot satisfy the requested E2 accuracy band: "
            f"successes={len(successes)}, failures={len(failures)}, target_size={n}"
        )
    desired = min(upper, max(lower, round(config.target_accuracy * n)))
    rng = random.Random(config.seed)
    rng.shuffle(successes)
    rng.shuffle(failures)
    selected = successes[:desired] + failures[: n - desired]
    rng.shuffle(selected)
    report = {
        "candidate_tasks": len(tasks),
        "candidate_e2_accuracy": len(successes) / len(tasks),
        "candidate_successes": len(successes),
        "candidate_failures": len(failures),
        "selected_tasks": n,
        "selected_successes": desired,
        "selected_failures": n - desired,
        "selected_e2_accuracy": desired / n,
        "min_accuracy": config.min_accuracy,
        "max_accuracy": config.max_accuracy,
        "target_accuracy": config.target_accuracy,
        "seed": config.seed,
    }
    return selected, report


EvaluateVariant = Callable[[E3ExperimentVariant], Mapping[str, Any]]


def run_variant_study(
    variants: Iterable[E3ExperimentVariant], evaluate: EvaluateVariant, output_dir: str,
) -> List[Dict[str, Any]]:
    """Evaluate variants through one metric schema and export plot-ready data."""
    rows = []
    for variant in variants:
        row = {**asdict(variant), **dict(evaluate(variant))}
        rows.append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "e3_variant_results.json").write_text(json.dumps(rows, indent=2, default=str))
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (output / "e3_variant_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return rows
