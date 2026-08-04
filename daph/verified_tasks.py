"""Deterministic multi-family tasks and leakage-resistant split construction."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
import random
from typing import Any, Dict, Mapping, Sequence


GENERATOR_VERSION = "daph_verified_multifamily_v2"
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
            # Template variation increases the number of independent grouped-
            # bootstrap clusters without changing the verified task itself.
            template_variant = index % 3
            if template_variant == 1:
                prompt = prompt.replace("\nAnswer:", "\nGive only the resulting integer:")
                template = template.removesuffix("_v1") + "_v2"
            elif template_variant == 2:
                prompt = "Solve the following exactly.\n" + prompt
                template = template.removesuffix("_v1") + "_v3"
            generator_bucket = _difficulty(scale)
            task_id = f"{family}-{seed}-{index}"
            tasks.append({
                "task_id": task_id,
                "prompt": prompt,
                "expected": str(expected),
                "task_family": family,
                "template_id": template,
                # This is a generator-scale bucket, deliberately not a claim
                # about empirical/model-defined reasoning difficulty.
                "difficulty": f"GENERATOR_{generator_bucket}",
                "difficulty_bucket": f"GENERATOR_{generator_bucket}",
                "difficulty_source": "generator_numeric_scale_v1",
                "empirical_difficulty": None,
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
    included_families: Sequence[str] | None = None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Select an E2 mixed-success sensitivity set without consulting E3."""
    if not 0 <= target_e2_accuracy <= 1:
        raise ValueError("target_e2_accuracy must be in [0, 1]")
    outcome_by_id = {str(row["task_id"]): bool(row["e2_correct"]) for row in e2_outcomes}
    if any(str(task["task_id"]) not in outcome_by_id for task in tasks):
        raise ValueError("Every calibration candidate needs an E2 outcome")
    if included_families is not None:
        allowed = {str(family) for family in included_families}
        tasks = [task for task in tasks if str(task["task_family"]) in allowed]
        if not tasks:
            raise ValueError("included_families selected no calibration tasks")
    if count < 1 or count > len(tasks):
        raise ValueError("Calibrated split count must be within the available task count")
    rng = random.Random(seed)
    by_family: Dict[str, Dict[bool, list[Mapping[str, Any]]]] = defaultdict(lambda: {True: [], False: []})
    for task in tasks:
        by_family[str(task["task_family"])][outcome_by_id[str(task["task_id"])]].append(task)
    families = sorted(by_family)
    # Equal allocation prevents a family with many E2 successes/failures from
    # becoming a proxy for correctness class in the calibrated set.
    base, extra = divmod(count, len(families))
    quotas = {family: base + int(index < extra) for index, family in enumerate(families)}
    ideal_successes = {family: quotas[family] * target_e2_accuracy for family in families}
    success_quotas = {family: int(ideal_successes[family]) for family in families}
    remaining_successes = round(count * target_e2_accuracy) - sum(success_quotas.values())
    for family in sorted(
        families, key=lambda key: (-(ideal_successes[key] - success_quotas[key]), key),
    )[:remaining_successes]:
        success_quotas[family] += 1
    selected: list[Dict[str, Any]] = []
    per_family: Dict[str, Dict[str, Any]] = {}
    for family in families:
        successes, failures = list(by_family[family][True]), list(by_family[family][False])
        wanted_successes = success_quotas[family]
        wanted_failures = quotas[family] - wanted_successes
        if len(successes) < wanted_successes or len(failures) < wanted_failures:
            raise ValueError(
                "Cannot form a family-stratified calibrated split; "
                f"family={family!r} needs {wanted_successes} E2 successes and {wanted_failures} failures, "
                f"has {len(successes)} successes and {len(failures)} failures"
            )
        rng.shuffle(successes)
        rng.shuffle(failures)
        chosen = successes[:wanted_successes] + failures[:wanted_failures]
        selected.extend(dict(task) for task in chosen)
        per_family[family] = {
            "available_e2_successes": len(successes),
            "available_e2_failures": len(failures),
            "selected_tasks": len(chosen),
            "selected_e2_successes": wanted_successes,
            "selected_e2_failures": wanted_failures,
            "selected_e2_accuracy": wanted_successes / len(chosen) if chosen else None,
        }
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
        "family_stratified": True,
        "family_allocation": "balanced_equal",
        "per_task_family": per_family,
        "included_families": families,
    }


def choose_calibration_families(
    tasks: Sequence[Mapping[str, Any]], e2_outcomes: Sequence[Mapping[str, Any]], *,
    split_counts: Sequence[int], target_e2_accuracy: float = 0.5,
    minimum_families: int = 5,
) -> tuple[tuple[str, ...], Dict[str, Any]]:
    """Choose the largest E2-mixed family subset by a predeclared capacity rule."""
    if not 0 <= target_e2_accuracy <= 1:
        raise ValueError("target_e2_accuracy must be in [0, 1]")
    if not split_counts or any(int(count) < 1 for count in split_counts):
        raise ValueError("split_counts must contain positive counts")
    outcome_by_id = {str(row["task_id"]): bool(row["e2_correct"]) for row in e2_outcomes}
    by_family: Dict[str, Dict[bool, int]] = defaultdict(lambda: {True: 0, False: 0})
    for task in tasks:
        task_id = str(task["task_id"])
        if task_id not in outcome_by_id:
            raise ValueError("Every calibration candidate needs an E2 outcome")
        by_family[str(task["task_family"])][outcome_by_id[task_id]] += 1
    families = sorted(by_family)
    if not 1 <= minimum_families <= len(families):
        raise ValueError("minimum_families must be within the available family count")

    def requirements(subset: tuple[str, ...]) -> Dict[str, Dict[bool, int]]:
        needed = {family: {True: 0, False: 0} for family in subset}
        for split_count in split_counts:
            base, extra = divmod(int(split_count), len(subset))
            quotas = {family: base + int(index < extra) for index, family in enumerate(subset)}
            ideals = {family: quotas[family] * target_e2_accuracy for family in subset}
            successes = {family: int(ideals[family]) for family in subset}
            remaining = round(int(split_count) * target_e2_accuracy) - sum(successes.values())
            for family in sorted(
                subset, key=lambda key: (-(ideals[key] - successes[key]), key),
            )[:remaining]:
                successes[family] += 1
            for family in subset:
                needed[family][True] += successes[family]
                needed[family][False] += quotas[family] - successes[family]
        return needed

    feasible: list[tuple[float, tuple[str, ...], Dict[str, Dict[bool, int]]]] = []
    for size in range(len(families), minimum_families - 1, -1):
        for subset in itertools.combinations(families, size):
            needed = requirements(subset)
            if all(
                by_family[family][outcome] >= needed[family][outcome]
                for family in subset for outcome in (True, False)
            ):
                # Prefer the subset with the largest minimum supply margin;
                # ties are deterministic. This inspects only E2 calibration
                # capacity, never E3 outcomes.
                margin = min(
                    by_family[family][outcome] - needed[family][outcome]
                    for family in subset for outcome in (True, False)
                )
                feasible.append((float(margin), subset, needed))
        if feasible:
            break
    if not feasible:
        raise ValueError(
            f"No family-stratified calibration subset with at least {minimum_families} families "
            "can supply the requested E2 success/failure counts"
        )
    _, selected, needed = max(feasible, key=lambda item: (item[0], item[1]))
    excluded = [family for family in families if family not in selected]
    return selected, {
        "selection_rule": "largest_feasible_family_subset_then_maximum_minimum_supply_margin",
        "selection_inputs": ["task_family", "e2_correct", "predeclared_split_counts"],
        "e3_outcomes_inspected": False,
        "minimum_families": minimum_families,
        "split_counts": [int(count) for count in split_counts],
        "included_families": list(selected),
        "excluded_families": excluded,
        "available_by_family": {
            family: {
                "e2_successes": by_family[family][True],
                "e2_failures": by_family[family][False],
            } for family in families
        },
        "required_by_included_family": {
            family: {
                "e2_successes": needed[family][True],
                "e2_failures": needed[family][False],
            } for family in selected
        },
    }
