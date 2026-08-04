"""Layer-contribution profiling inspired by Zhang et al. (arXiv:2607.01232v2).

The paper is a structural prior, not evidence for this model. This subsystem
measures contribution on the exact checkpoint and preserves negative results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .counterfactual import full_state_dict_digest
from .e3_architecture import (
    middle_only_profile_indices, select_profiled_layers, sparse_profile_indices,
)


@dataclass(frozen=True)
class LayerAdaptationObjective:
    kind: str = "supervised_ce"
    name: str = "causal_lm_ce"
    verified_reward: bool = False
    external_callback_name: Optional[str] = None

    def validate(self) -> None:
        allowed = {"supervised_ce", "verified_reward", "external_callback"}
        if self.kind not in allowed:
            raise ValueError(f"Unsupported layer adaptation objective: {self.kind}")
        if self.kind == "verified_reward" and not self.verified_reward:
            raise ValueError("Verified-reward objectives must explicitly set verified_reward=True")
        if self.kind == "external_callback" and not self.external_callback_name:
            raise ValueError("External objectives must record their callback name")


@dataclass
class LayerContributionConfig:
    profile_mode: str = "sparse"
    training_steps: int = 100
    seed: int = 42
    epsilon: float = 1e-8
    explicit_layers: Optional[List[int]] = None
    best_contiguous_width: int = 3
    objective: LayerAdaptationObjective = field(default_factory=LayerAdaptationObjective)
    validation_metric: str = "verified_accuracy"
    model_source_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    def layers(self, num_layers: int) -> Tuple[int, ...]:
        if self.explicit_layers is not None:
            layers = tuple(sorted(set(self.explicit_layers)))
        elif self.profile_mode == "full":
            layers = tuple(range(num_layers))
        elif self.profile_mode == "sparse":
            layers = sparse_profile_indices(num_layers)
        elif self.profile_mode == "middle_only":
            layers = middle_only_profile_indices(num_layers)
        else:
            raise ValueError(f"Unsupported profile_mode: {self.profile_mode}")
        if any(layer < 0 or layer >= num_layers for layer in layers):
            raise ValueError("Profile layer outside model depth")
        return layers


@dataclass(frozen=True)
class LayerContributionResult:
    layer_index: int
    relative_depth: float
    score_base: float
    score_full: float
    score_layer: float
    layer_contribution: float
    training_steps: int
    trainable_parameter_count: int
    validation_metric: str
    seed: int
    checkpoint_digest: str


@dataclass
class LayerContributionReport:
    profile_status: str
    model_digest: str
    config_digest: str
    score_base: float
    score_full: float
    results: List[LayerContributionResult]
    ranking: List[int]
    best_contiguous_region: List[int]
    mean_by_depth_quartile: Dict[str, float]
    mean_middle_40_60: Optional[float]
    depth_correlation: Optional[float]
    middle_concentration_observed: bool

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


EvaluateFn = Callable[[torch.nn.Module], float]
AdaptFn = Callable[[torch.nn.Module, Optional[int], LayerAdaptationObjective, int, int], None]


def profile_selection_payload(
    report: LayerContributionReport, *, strategy: str = "best_contiguous",
    top_k: int = 3, contiguous_width: int = 3, threshold: float = 0.5,
) -> Dict[str, Any]:
    """Create the immutable runtime selection fields from a measured profile."""
    contributions = {result.layer_index: result.layer_contribution for result in report.results}
    selected = select_profiled_layers(
        contributions, strategy=strategy, top_k=top_k,
        contiguous_width=contiguous_width, threshold=threshold,
    )
    return {
        "e3_region_selection": "profiled",
        "e3_profiled_layers": list(selected),
        "profile_selection_strategy": strategy,
        "source_profile_digest": report.digest(),
        "profile_status": report.profile_status,
    }


class LayerContributionProfiler:
    def __init__(self, model: torch.nn.Module, config: LayerContributionConfig) -> None:
        if not hasattr(model, "layers"):
            raise ValueError("Layer profiling requires model.layers")
        config.objective.validate()
        self.model = model
        self.config = config
        self.layers = config.layers(len(model.layers))

    @staticmethod
    def _set_only_layer_trainable(model: torch.nn.Module, layer_index: int) -> int:
        count = 0
        prefix = f"layers.{layer_index}.base."
        for name, parameter in model.named_parameters():
            enabled = name.startswith(prefix)
            parameter.requires_grad_(enabled)
            if enabled:
                count += parameter.numel()
        return count

    def run(
        self, evaluate_fn: EvaluateFn, adapt_fn: AdaptFn,
        *, full_reference_adapter: Optional[AdaptFn] = None,
        score_full: Optional[float] = None,
    ) -> LayerContributionReport:
        base_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
        base_flags = [parameter.requires_grad for parameter in self.model.parameters()]
        model_digest = full_state_dict_digest(self.model)
        score_base = float(evaluate_fn(self.model))
        if score_full is None:
            if full_reference_adapter is None:
                raise ValueError("Provide score_full or a full_reference_adapter")
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            full_reference_adapter(self.model, None, self.config.objective, self.config.training_steps, self.config.seed)
            score_full = float(evaluate_fn(self.model))
            self.model.load_state_dict(base_state)
        reference_gain = float(score_full) - score_base
        if not math.isfinite(reference_gain) or reference_gain <= self.config.epsilon:
            self.model.load_state_dict(base_state)
            for parameter, flag in zip(self.model.parameters(), base_flags):
                parameter.requires_grad_(flag)
            raise ValueError(
                "Full-reference adaptation did not improve the baseline by more than epsilon; "
                f"score_base={score_base}, score_full={score_full}, epsilon={self.config.epsilon}. "
                "Layer contributions are not qualified for normalization or selection."
            )
        denominator = reference_gain
        results: List[LayerContributionResult] = []
        for layer_index in self.layers:
            self.model.load_state_dict(base_state)
            trainable = self._set_only_layer_trainable(self.model, layer_index)
            adapt_fn(self.model, layer_index, self.config.objective, self.config.training_steps, self.config.seed)
            score_layer = float(evaluate_fn(self.model))
            contribution = (score_layer - score_base) / denominator
            results.append(LayerContributionResult(
                layer_index=layer_index,
                relative_depth=layer_index / max(len(self.model.layers) - 1, 1),
                score_base=score_base,
                score_full=float(score_full),
                score_layer=score_layer,
                layer_contribution=contribution,
                training_steps=self.config.training_steps,
                trainable_parameter_count=trainable,
                validation_metric=self.config.validation_metric,
                seed=self.config.seed,
                checkpoint_digest=full_state_dict_digest(self.model),
            ))
        self.model.load_state_dict(base_state)
        for parameter, flag in zip(self.model.parameters(), base_flags):
            parameter.requires_grad_(flag)
        return self._summarize(model_digest, score_base, float(score_full), results)

    def _summarize(self, model_digest: str, score_base: float, score_full: float, results: List[LayerContributionResult]) -> LayerContributionReport:
        ranking = [r.layer_index for r in sorted(results, key=lambda item: (-item.layer_contribution, item.layer_index))]
        contributions = {r.layer_index: r.layer_contribution for r in results}
        width = min(max(1, self.config.best_contiguous_width), len(results))
        candidates = []
        for start in range(0, len(self.model.layers) - width + 1):
            region = list(range(start, start + width))
            if all(index in contributions for index in region):
                candidates.append((sum(contributions[index] for index in region) / width, region))
        best_region = max(candidates, key=lambda item: (item[0], [-x for x in item[1]]))[1] if candidates else []
        quartiles: Dict[str, List[float]] = {f"Q{i}": [] for i in range(1, 5)}
        middle = []
        for result in results:
            quartile = min(3, int(result.relative_depth * 4)) + 1
            quartiles[f"Q{quartile}"].append(result.layer_contribution)
            if 0.4 <= result.relative_depth <= 0.6:
                middle.append(result.layer_contribution)
        means = {key: (sum(values) / len(values) if values else float("nan")) for key, values in quartiles.items()}
        correlation = _pearson([r.relative_depth for r in results], [r.layer_contribution for r in results])
        middle_mean = sum(middle) / len(middle) if middle else None
        outer = [r.layer_contribution for r in results if not 0.4 <= r.relative_depth <= 0.6]
        observed = bool(middle and outer and middle_mean > sum(outer) / len(outer))
        config_json = json.dumps(asdict(self.config), sort_keys=True, default=str)
        return LayerContributionReport(
            profile_status="FULL_PROFILE" if len(results) == len(self.model.layers) else "PARTIAL_PROFILE",
            model_digest=model_digest,
            config_digest=hashlib.sha256(config_json.encode()).hexdigest(),
            score_base=score_base, score_full=score_full, results=results,
            ranking=ranking, best_contiguous_region=best_region,
            mean_by_depth_quartile=means, mean_middle_40_60=middle_mean,
            depth_correlation=correlation, middle_concentration_observed=observed,
        )

    def save(self, report: LayerContributionReport, output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(json.dumps({
            "model_digest": report.model_digest, "config_digest": report.config_digest,
            "profile_digest": report.digest(), "profile_status": report.profile_status,
            "config": asdict(self.config),
        }, indent=2, default=str))
        (output / "environment.json").write_text(json.dumps({
            "python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
        }, indent=2))
        (output / "base_metrics.json").write_text(json.dumps({"score": report.score_base}, indent=2))
        (output / "full_reference_metrics.json").write_text(json.dumps({"score": report.score_full}, indent=2))
        with (output / "per_layer_results.jsonl").open("w") as handle:
            for result in report.results:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
        (output / "rankings.json").write_text(json.dumps({
            "ranking": report.ranking, "best_contiguous_region": report.best_contiguous_region,
        }, indent=2))
        with (output / "layer_contribution.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(report.results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(result) for result in report.results)
        top = report.ranking[:5]
        bottom = list(reversed(report.ranking[-5:]))
        (output / "summary.md").write_text(
            "# Layer contribution profile\n\n"
            f"Status: **{report.profile_status}**\n\n"
            f"Top layers: {top}\n\nBottom layers: {bottom}\n\n"
            f"Best contiguous region: {report.best_contiguous_region}\n\n"
            f"Mean 40%-60% contribution: {report.mean_middle_40_60}\n\n"
            f"Depth correlation: {report.depth_correlation}\n\n"
            f"Middle concentration observed: {report.middle_concentration_observed}\n"
        )


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None
