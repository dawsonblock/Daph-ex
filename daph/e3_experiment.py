"""Unified E3 variant, location, and dose-response experiment contracts."""

from __future__ import annotations

import csv
import json
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
