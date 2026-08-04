"""Configuration and deterministic layer selection for canonical E3 variants."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


E3_MODES = frozenset({
    "none", "final_refine", "middle_recurrent", "middle_repeat",
    "profiled_middle_recurrent",
})
REGION_SELECTION_MODES = frozenset({"fraction", "middle_heuristic", "profiled", "manual"})
PROFILE_SELECTION_STRATEGIES = frozenset({"top_k", "best_contiguous", "threshold"})


@dataclass
class E3RefinementConfig:
    """Serializable E3 architecture contract.

    Layer indices are zero-based. Fractional regions include every layer from
    ``floor(start * L)`` through ``ceil(end * L) - 1``.
    """

    e3_refinement_mode: str = "middle_recurrent"
    e3_region_start_fraction: float = 0.40
    e3_region_end_fraction: float = 0.60
    e3_refine_steps: int = 2
    e3_max_refine_steps: int = 8
    e3_reuse_pretrained_layers: bool = False
    e3_train_middle_layers: bool = False
    e3_refinement_scale_init: float = 0.0
    e3_training_scale_epsilon: float = 1e-4
    e3_profiled_layers: Optional[List[int]] = None
    e3_region_selection: str = "middle_heuristic"
    e3_allow_layer_weight_updates: bool = False
    e3_allow_refiner_updates: bool = True
    e3_hard_case_sampling_ratio: float = 0.80
    e3_regression_guard_weight: float = 0.05
    e3_repeat_count: int = 1
    e3_reuse_layers: Optional[List[int]] = None
    profile_selection_strategy: str = "best_contiguous"
    profile_top_k: int = 3
    profile_contiguous_width: int = 3
    profile_contribution_threshold: float = 0.5
    source_profile_digest: Optional[str] = None

    def validate(self, num_layers: int) -> None:
        if self.e3_refinement_mode not in E3_MODES:
            raise ValueError(f"Unsupported E3 mode: {self.e3_refinement_mode}")
        if self.e3_region_selection not in REGION_SELECTION_MODES:
            raise ValueError(f"Unsupported E3 region selection: {self.e3_region_selection}")
        if self.profile_selection_strategy not in PROFILE_SELECTION_STRATEGIES:
            raise ValueError(f"Unsupported profile selection strategy: {self.profile_selection_strategy}")
        if not 0 <= self.e3_region_start_fraction < self.e3_region_end_fraction <= 1:
            raise ValueError("E3 region fractions must satisfy 0 <= start < end <= 1")
        minimum_steps = 0 if self.e3_refinement_mode == "none" else 1
        if not minimum_steps <= self.e3_refine_steps <= self.e3_max_refine_steps:
            raise ValueError(f"e3_refine_steps must be within [{minimum_steps}, e3_max_refine_steps]")
        if self.e3_refinement_scale_init != 0.0:
            raise ValueError("Gate 0B requires e3_refinement_scale_init=0; use e3_training_scale_epsilon after the gate")
        if self.e3_repeat_count < 1:
            raise ValueError("e3_repeat_count must be positive")
        for layer in (self.e3_profiled_layers or []) + (self.e3_reuse_layers or []):
            if layer < 0 or layer >= num_layers:
                raise ValueError(f"Layer index {layer} outside [0, {num_layers - 1}]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class E3RegionSelection:
    selection_method: str
    selected_layers: Tuple[int, ...]
    insertion_layer: int
    region_start: int
    region_end: int
    source_profile_digest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fractional_region(num_layers: int, start: float, end: float) -> Tuple[int, ...]:
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    first = max(0, min(num_layers - 1, int(math.floor(start * num_layers))))
    stop = max(first + 1, min(num_layers, int(math.ceil(end * num_layers))))
    return tuple(range(first, stop))


def sparse_profile_indices(num_layers: int) -> Tuple[int, ...]:
    fractions = (0.0, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 1.0)
    return tuple(sorted({min(num_layers - 1, int(round(f * (num_layers - 1)))) for f in fractions}))


def middle_only_profile_indices(num_layers: int) -> Tuple[int, ...]:
    return fractional_region(num_layers, 0.30, 0.70)


def select_profiled_layers(
    contributions: Dict[int, float], *, strategy: str, top_k: int = 3,
    contiguous_width: int = 3, threshold: float = 0.5,
) -> Tuple[int, ...]:
    if not contributions:
        raise ValueError("Profile contributions are required")
    if strategy == "top_k":
        return tuple(sorted(k for k, _ in sorted(contributions.items(), key=lambda item: (-item[1], item[0]))[:top_k]))
    if strategy == "threshold":
        selected = tuple(sorted(k for k, value in contributions.items() if value >= threshold))
        if not selected:
            raise ValueError("No layers meet the configured contribution threshold")
        return selected
    if strategy == "best_contiguous":
        keys = sorted(contributions)
        width = min(max(1, contiguous_width), len(keys))
        candidates = []
        for start in range(0, len(keys) - width + 1):
            region = tuple(keys[start:start + width])
            if region[-1] - region[0] != width - 1:
                continue
            candidates.append((sum(contributions[k] for k in region) / width, region))
        if not candidates:
            raise ValueError("Profile has no contiguous region of the configured width")
        return max(candidates, key=lambda item: (item[0], tuple(-k for k in item[1])))[1]
    raise ValueError(f"Unknown profile selection strategy: {strategy}")


def resolve_e3_region(
    config: E3RefinementConfig, num_layers: int,
    contributions: Optional[Dict[int, float]] = None,
) -> E3RegionSelection:
    config.validate(num_layers)
    if config.e3_region_selection in {"fraction", "middle_heuristic"}:
        layers = fractional_region(num_layers, config.e3_region_start_fraction, config.e3_region_end_fraction)
        method = config.e3_region_selection
    elif config.e3_region_selection == "manual":
        layers = tuple(sorted(set(config.e3_profiled_layers or [])))
        method = "manual"
    else:
        if not config.source_profile_digest:
            raise ValueError("Profile-guided E3 selection requires source_profile_digest")
        if config.e3_profiled_layers:
            # Persisted profiles normally resolve the strategy before model
            # construction; the selected indices and digest are the immutable
            # runtime contract.
            layers = tuple(sorted(set(config.e3_profiled_layers)))
            method = f"profiled:preselected:{config.profile_selection_strategy}"
        else:
            layers = select_profiled_layers(
                contributions or {}, strategy=config.profile_selection_strategy,
                top_k=config.profile_top_k, contiguous_width=config.profile_contiguous_width,
                threshold=config.profile_contribution_threshold,
            )
            method = f"profiled:{config.profile_selection_strategy}"
    if not layers:
        raise ValueError("Resolved E3 region is empty")
    insertion = layers[len(layers) // 2]
    return E3RegionSelection(method, layers, insertion, min(layers), max(layers), config.source_profile_digest)
