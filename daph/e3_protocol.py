"""Reproducible E3 experiment tiers, profile stability, and evidence metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


class ExperimentTier(str, Enum):
    SMOKE = "SMOKE"
    PILOT = "PILOT"
    QUALIFICATION = "QUALIFICATION"
    FINAL = "FINAL"


class ProfileTier(str, Enum):
    PROFILE_SMOKE = "PROFILE_SMOKE"
    PROFILE_PILOT = "PROFILE_PILOT"
    PROFILE_FULL = "PROFILE_FULL"


PROFILE_TIER_REQUIREMENTS = {
    ProfileTier.PROFILE_SMOKE: {"training_examples": 8, "validation_examples": 4, "seeds": 1, "updates": 2},
    ProfileTier.PROFILE_PILOT: {"training_examples": 200, "validation_examples": 200, "seeds": 3, "updates": 20},
    ProfileTier.PROFILE_FULL: {"training_examples": 1000, "validation_examples": 500, "seeds": 3, "updates": 100},
}


def validate_profile_tier(
    tier: ProfileTier, *, training_examples: int, validation_examples: int,
    seeds: int, updates: int,
) -> Dict[str, Any]:
    requirement = PROFILE_TIER_REQUIREMENTS[tier]
    observed = {
        "training_examples": training_examples, "validation_examples": validation_examples,
        "seeds": seeds, "updates": updates,
    }
    passed = all(observed[key] >= minimum for key, minimum in requirement.items())
    return {"tier": tier.value, "passed": passed, "minimum": requirement, "observed": observed}


class ClaimStrength(str, Enum):
    ENGINEERING_PASS = "ENGINEERING_PASS"
    MECHANISM_SIGNAL = "MECHANISM_SIGNAL"
    PILOT_EVIDENCE = "PILOT_EVIDENCE"
    STATISTICALLY_QUALIFIED = "STATISTICALLY_QUALIFIED"
    FINAL_EVIDENCE = "FINAL_EVIDENCE"


@dataclass(frozen=True)
class ExperimentScale:
    tier: ExperimentTier
    heldout_examples: int
    training_seeds: tuple[int, ...]
    evaluation_seed: int
    predeclared: bool = True

    def validate(self) -> None:
        if self.heldout_examples < 1:
            raise ValueError("heldout_examples must be positive")
        if not self.training_seeds:
            raise ValueError("At least one independent training seed is required")
        if self.tier == ExperimentTier.QUALIFICATION and self.heldout_examples < 500:
            raise ValueError("QUALIFICATION requires at least 500 held-out examples")
        if self.tier == ExperimentTier.FINAL and not self.predeclared:
            raise ValueError("FINAL sample size must be predeclared")


def _rank(values: Mapping[int, float]) -> Dict[int, float]:
    ordered = sorted(values, key=lambda key: (values[key], key))
    ranks: Dict[int, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        average = (index + 1 + end) / 2.0
        for key in ordered[index:end]:
            ranks[key] = average
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    ml, mr = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - ml) * (b - mr) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - ml) ** 2 for a in left) * sum((b - mr) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def _best_region(values: Mapping[int, float], width: int) -> tuple[int, ...]:
    layers = sorted(values)
    candidates = []
    for start in layers:
        region = tuple(range(start, start + width))
        if all(layer in values for layer in region):
            candidates.append((sum(values[layer] for layer in region) / width, region))
    return max(candidates, key=lambda item: (item[0], tuple(-x for x in item[1])))[1] if candidates else ()


def profile_stability(
    profiles: Mapping[int, Mapping[int, float]], *, top_k: int = 3,
    contiguous_width: int = 3, middle_fraction: tuple[float, float] = (0.4, 0.6),
    min_spearman: float = 0.6, min_top_k_overlap: float = 0.5,
) -> Dict[str, Any]:
    """Measure ranking/location reproducibility across profile seeds."""
    if len(profiles) < 2:
        raise ValueError("Profile stability requires at least two seeds")
    seeds = sorted(profiles)
    shared = set(profiles[seeds[0]])
    for seed in seeds[1:]:
        shared &= set(profiles[seed])
    if len(shared) < 2:
        raise ValueError("Profile seeds need at least two shared layers")
    layers = sorted(shared)
    pairwise_spearman, top_overlaps = [], []
    rankings: Dict[int, list[int]] = {}
    regions: Dict[int, tuple[int, ...]] = {}
    for seed in seeds:
        rankings[seed] = sorted(layers, key=lambda layer: (-float(profiles[seed][layer]), layer))
        regions[seed] = _best_region(profiles[seed], contiguous_width)
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1:]:
            left_rank, right_rank = _rank(profiles[left_seed]), _rank(profiles[right_seed])
            pairwise_spearman.append(_pearson(
                [left_rank[layer] for layer in layers], [right_rank[layer] for layer in layers],
            ))
            left_top, right_top = set(rankings[left_seed][:top_k]), set(rankings[right_seed][:top_k])
            top_overlaps.append(len(left_top & right_top) / max(len(left_top | right_top), 1))
    best_layers = [ranking[0] for ranking in rankings.values()]
    best_regions = list(regions.values())
    max_layer = max(layers)
    middle_by_seed = []
    for seed in seeds:
        middle = [layer for layer in layers if middle_fraction[0] <= layer / max(max_layer, 1) <= middle_fraction[1]]
        outer = [layer for layer in layers if layer not in middle]
        middle_by_seed.append(bool(
            middle and outer
            and sum(profiles[seed][layer] for layer in middle) / len(middle)
            > sum(profiles[seed][layer] for layer in outer) / len(outer)
        ))
    mean_spearman = sum(pairwise_spearman) / len(pairwise_spearman)
    mean_overlap = sum(top_overlaps) / len(top_overlaps)
    stable = mean_spearman >= min_spearman and mean_overlap >= min_top_k_overlap
    contribution_summary = {
        str(layer): {
            "mean": sum(float(profiles[seed][layer]) for seed in seeds) / len(seeds),
            "std": (
                math.sqrt(sum((float(profiles[seed][layer]) - sum(float(profiles[s][layer]) for s in seeds) / len(seeds)) ** 2 for seed in seeds) / (len(seeds) - 1))
                if len(seeds) > 1 else 0.0
            ),
        }
        for layer in layers
    }
    return {
        "seeds": seeds,
        "shared_layers": layers,
        "pairwise_spearman": pairwise_spearman,
        "mean_spearman": mean_spearman,
        "top_k": top_k,
        "pairwise_top_k_overlap": top_overlaps,
        "mean_top_k_overlap": mean_overlap,
        "best_layers": best_layers,
        "best_layer_stability": max(best_layers.count(layer) for layer in set(best_layers)) / len(best_layers),
        "best_contiguous_regions": [list(region) for region in best_regions],
        "best_region_stability": max(best_regions.count(region) for region in set(best_regions)) / len(best_regions),
        "middle_region_stability": sum(middle_by_seed) / len(middle_by_seed),
        "contribution_by_layer": contribution_summary,
        "stable_for_promotion": stable,
        "thresholds": {"min_spearman": min_spearman, "min_top_k_overlap": min_top_k_overlap},
    }


def promote_e3_placement(
    candidates: Sequence[Mapping[str, Any]], *, profile_stable: bool,
) -> Dict[str, Any]:
    """Promote only replicated, cost-effective candidates; prefer heuristic on unstable profiles."""
    if not candidates:
        raise ValueError("At least one placement candidate is required")
    eligible = []
    for candidate in candidates:
        required = ("name", "quality_lcb95", "utility_lcb95", "rescues", "regressions", "seed_pass_rate", "compute_delta")
        missing = [field for field in required if field not in candidate]
        if missing:
            raise ValueError(f"Placement candidate is missing: {', '.join(missing)}")
        profile_candidate = str(candidate["name"]).startswith("PROFILED")
        if (
            float(candidate["quality_lcb95"]) > 0
            and float(candidate["utility_lcb95"]) > 0
            and int(candidate["rescues"]) > int(candidate["regressions"])
            and float(candidate["seed_pass_rate"]) >= 2 / 3
            and (not profile_candidate or profile_stable)
        ):
            eligible.append(candidate)
    if not eligible:
        fallback = "HEURISTIC_MIDDLE" if any(item["name"] == "HEURISTIC_MIDDLE" for item in candidates) else "FINAL"
        return {
            "promoted": False, "canonical": fallback,
            "reason": "NO_REPLICATED_QUALITY_AND_UTILITY_PASS",
            "profile_stable": profile_stable,
        }
    winner = max(eligible, key=lambda item: (
        float(item["utility_lcb95"]), float(item["quality_lcb95"]),
        int(item["rescues"]) - int(item["regressions"]),
        -float(item["compute_delta"]), str(item["name"]),
    ))
    return {
        "promoted": True, "canonical": winner["name"],
        "reason": "PREDECLARED_PROMOTION_RULE_PASS",
        "profile_stable": profile_stable,
    }


@dataclass(frozen=True)
class EvidenceMetadata:
    artifact_commit: str
    repository_version: str
    test_count_at_creation: int
    pytest_digest: str
    config_digest: str
    source_tree_digest: str
    claim_strength: ClaimStrength
    created_at: str = ""

    def normalized(self) -> Dict[str, Any]:
        values = asdict(self)
        values["claim_strength"] = self.claim_strength.value
        values["created_at"] = self.created_at or datetime.now(timezone.utc).isoformat()
        for field in ("artifact_commit", "repository_version", "pytest_digest", "config_digest", "source_tree_digest"):
            if not str(values[field]).strip():
                raise ValueError(f"Evidence metadata requires {field}")
        return values


def digest_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_evidence_metadata(output_dir: str | Path, metadata: EvidenceMetadata) -> Path:
    """Create immutable artifact metadata; never rewrite historical evidence."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "artifact_metadata.json"
    if destination.exists():
        raise FileExistsError(f"Historical artifact metadata already exists: {destination}")
    destination.write_text(json.dumps(metadata.normalized(), indent=2, sort_keys=True) + "\n")
    return destination
