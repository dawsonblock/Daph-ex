"""Staged E3 training controls and verified-task objective hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .qwen_exfusion import QwenExFusionModel, TrainingInitReceipt, prepare_exfusion_for_training


@dataclass(frozen=True)
class E3StageConfig:
    stage: str = "E3-A"
    train_refiner: bool = True
    train_selected_middle_layers: bool = False
    pretrained_middle_lr: float = 1e-5
    refiner_lr: float = 1e-4
    scale_lr: float = 1e-3
    regression_guard_weight: float = 0.05
    objective_name: str = "external_verified_task"

    def validate(self) -> None:
        if self.stage not in {"E3-A", "E3-B", "E3-C"}:
            raise ValueError("stage must be E3-A, E3-B, or E3-C")
        if self.regression_guard_weight < 0:
            raise ValueError("regression_guard_weight must be non-negative")


VerifiedTaskLossFn = Callable[[Mapping[str, Any], Mapping[str, Any]], Tensor]


def configure_e3_training(
    model: QwenExFusionModel, config: E3StageConfig,
) -> Tuple[TrainingInitReceipt, Dict[str, Tuple[str, ...]]]:
    """Freeze E2 and open only the predeclared E3-A/B parameter sets."""
    config.validate()
    provenance = model.parameter_provenance
    if provenance is None:
        raise RuntimeError("Exact parameter provenance is required for E3 training")
    refiner = set(provenance.e3_refinement_parameter_names) if (
        config.train_refiner and model.e3_config.e3_allow_refiner_updates
    ) else set()
    scales = set(provenance.e3_scale_parameter_names)
    middle = set(provenance.e3_middle_layer_parameter_names) if (
        config.train_selected_middle_layers
        and model.e3_config.e3_train_middle_layers
        and model.e3_config.e3_allow_layer_weight_updates
    ) else set()
    enabled = refiner | scales | middle
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in enabled)
    receipt = prepare_exfusion_for_training(
        model, gate0b_passed=True, epsilon=model.e3_config.e3_training_scale_epsilon,
    )
    return receipt, {
        "refiner": tuple(sorted(refiner)), "scales": tuple(sorted(scales)),
        "middle_layers": tuple(sorted(middle)), "frozen": tuple(sorted(
            name for name, _ in model.named_parameters() if name not in enabled
        )),
    }


def e3_verified_objective(
    e3_output: Mapping[str, Any], e2_output: Mapping[str, Any], task: Mapping[str, Any],
    verified_task_loss_fn: VerifiedTaskLossFn, *, regression_guard_weight: float,
) -> Tuple[Tensor, Dict[str, float]]:
    """Task-first E3 loss with a deliberately secondary E2 regression guard."""
    task_loss = verified_task_loss_fn(e3_output, task)
    if not isinstance(task_loss, Tensor) or not task_loss.requires_grad:
        raise ValueError("verified_task_loss_fn must return a differentiable scalar tensor")
    e3_logits = e3_output["logits"]
    e2_logits = e2_output["logits"]
    guard = F.kl_div(
        F.log_softmax(e3_logits.float(), dim=-1),
        F.softmax(e2_logits.detach().float(), dim=-1),
        reduction="batchmean",
    )
    total = task_loss + float(regression_guard_weight) * guard
    return total, {
        "verified_task_loss": float(task_loss.detach()),
        "regression_guard_kl": float(guard.detach()),
        "regression_guard_weight": float(regression_guard_weight),
    }
