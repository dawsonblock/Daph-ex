"""Deterministic multi-family tasks and leakage-resistant split construction."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random
from typing import Any, Dict, Mapping, Sequence


GENERATOR_VERSION = "daph_verified_multifamily_v1"
VERIFIER_VERSION = "exact_numeric_v1"


def _difficulty(scale: int) -> str:
    return "EASY" if scale < 20 else "MEDIUM" if scale < 100 else "HARD"


def generate_verified_tasks(*, count_per_family: int, seed: int) -> list[Dict[str, Any]]:
    """Generate exact-answer families without model-dependent selection."""
    if count_per_family < 1:
        raise ValueError("count_per_family must be positive")
    rng = random.Random(seed)
    tasks: list[Dict[str, Any]] = []
    families = (
        "addition_with_carry", "subtraction", "multiplication", "integer_comparison",
        "modular_arithmetic", "multi_step_arithmetic", "symbolic_substitution",
        "pattern_continuation", "code_output",
    )
    for family in families:
        for index in range(count_per_family):
            scale = (10, 99, 999)[index % 3]
            a, b = rng.randint(1, scale), rng.randint(1, scale)
            if family == "addition_with_carry":
                a = max(a, 10); b = max(b, 10)
                prompt, expected, template = f"Compute exactly: {a} + {b}\nAnswer:", a + b, "add_exact_v1"
            elif family == "subtraction":
                high, low = max(a, b), min(a, b)
                prompt, expected, template = f"Compute exactly: {high} - {low}\nAnswer:", high - low, "subtract_exact_v1"
            elif family == "multiplication":
                a, b = max(2, a % 40), max(2, b % 40)
                prompt, expected, template = f"Compute exactly: {a} * {b}\nAnswer:", a * b, "multiply_exact_v1"
            elif family == "integer_comparison":
                prompt = f"Return 1 if {a} is greater than {b}, -1 if smaller, and 0 if equal.\nAnswer:"
                expected, template = (1 if a > b else -1 if a < b else 0), "compare_integer_v1"
            elif family == "modular_arithmetic":
                modulus = rng.randint(2, 19)
                prompt, expected, template = f"Compute the non-negative remainder: {a} mod {modulus}\nAnswer:", a % modulus, "mod_exact_v1"
            elif family == "multi_step_arithmetic":
                c = rng.randint(1, max(2, scale // 4))
                prompt, expected, template = f"Compute exactly: ({a} + {b}) - {c}\nAnswer:", a + b - c, "two_step_add_sub_v1"
            elif family == "symbolic_substitution":
                x, coefficient, offset = rng.randint(1, 30), rng.randint(2, 9), rng.randint(-10, 10)
                prompt = f"If f(x) = {coefficient}x + ({offset}), compute f({x}).\nAnswer:"
                expected, template = coefficient * x + offset, "linear_substitution_v1"
            elif family == "pattern_continuation":
                start, step = rng.randint(0, 50), rng.randint(2, 15)
                sequence = [start + step * position for position in range(4)]
                prompt = f"Give the next integer: {', '.join(map(str, sequence))}, ?\nAnswer:"
                expected, template = start + 4 * step, "arithmetic_sequence_v1"
            else:
                loops = rng.randint(2, 8)
                prompt = f"What integer does this Python code print?\nx = {a}\nfor _ in range({loops}):\n    x += {b}\nprint(x)\nAnswer:"
                expected, template = a + loops * b, "python_loop_output_v1"
            task_id = f"{family}-{seed}-{index}"
            tasks.append({
                "task_id": task_id,
                "prompt": prompt,
                "expected": str(expected),
                "task_family": family,
                "template_id": template,
                "difficulty": _difficulty(scale),
                "difficulty_bucket": _difficulty(scale),
                "generator_version": GENERATOR_VERSION,
                "verifier_version": VERIFIER_VERSION,
                "generation_seed": seed,
            })
    return tasks


def natural_heldout_split(
    tasks: Sequence[Mapping[str, Any]], *, count: int, seed: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Select an untouched natural split; this API cannot inspect model outcomes."""
    if count < 1 or count > len(tasks):
        raise ValueError("Natural split count must be within the available task count")
    rng = random.Random(seed)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    selected = [dict(tasks[index]) for index in indices[:count]]
    ids = [str(task["task_id"]) for task in selected]
    return selected, {
        "split_type": "NATURAL_HELDOUT",
        "selection_inputs": ["task_id", "generator_seed"],
        "e2_outcomes_inspected": False,
        "e3_outcomes_inspected": False,
        "count": len(selected),
        "seed": seed,
        "task_ids_digest": hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
    }


def calibrated_sensitivity_split(
    tasks: Sequence[Mapping[str, Any]], e2_outcomes: Sequence[Mapping[str, Any]], *,
    count: int, target_e2_accuracy: float = 0.5, seed: int,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Select an E2 mixed-success sensitivity set without consulting E3."""
    if not 0 <= target_e2_accuracy <= 1:
        raise ValueError("target_e2_accuracy must be in [0, 1]")
    outcome_by_id = {str(row["task_id"]): bool(row["e2_correct"]) for row in e2_outcomes}
    if any(str(task["task_id"]) not in outcome_by_id for task in tasks):
        raise ValueError("Every calibration candidate needs an E2 outcome")
    rng = random.Random(seed)
    by_family: Dict[str, Dict[bool, list[Mapping[str, Any]]]] = defaultdict(lambda: {True: [], False: []})
    for task in tasks:
        by_family[str(task["task_family"])][outcome_by_id[str(task["task_id"])]].append(task)
    desired_successes = round(count * target_e2_accuracy)
    desired_failures = count - desired_successes
    successes = [task for buckets in by_family.values() for task in buckets[True]]
    failures = [task for buckets in by_family.values() for task in buckets[False]]
    if len(successes) < desired_successes or len(failures) < desired_failures:
        raise ValueError("Insufficient E2 successes/failures for the requested calibrated band")
    rng.shuffle(successes); rng.shuffle(failures)
    selected = [dict(task) for task in successes[:desired_successes] + failures[:desired_failures]]
    rng.shuffle(selected)
    return selected, {
        "split_type": "CALIBRATED_SENSITIVITY",
        "selection_inputs": ["task_id", "task_family", "e2_correct"],
        "e2_outcomes_inspected": True,
        "e3_outcomes_inspected": False,
        "count": len(selected),
        "selected_e2_accuracy": sum(outcome_by_id[str(task["task_id"])] for task in selected) / len(selected),
        "target_e2_accuracy": target_e2_accuracy,
        "seed": seed,
    }
