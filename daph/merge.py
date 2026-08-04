"""
Architecture-aware DARE → TIES → Fisher merge pipeline for DAPH / ExFusion v3.

Preserves the critical v2.3 corrections:
  - TIES pure sign-majority (heavy outlier cannot overturn weak majority)
  - Fisher denominator excludes DARE-dropped elements via keep masks
  - SSM-core policy can reduce drop rate / force soft merge
  - Optional torch.Generator for deterministic DARE
  - Independent family merges (e.g. SSM family vs attention/MoE family)

Works on arbitrary nn.Modules (SelectiveSSM, LatentMoE, HybridBlock, full model).
"""

from __future__ import annotations

import copy
import math
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

StructuredBatch = Union[Tensor, Sequence[Any], Mapping[str, Any]]
LossFn = Callable[[Any, Any], Tensor]
ForwardFn = Callable[[nn.Module, Any], Any]


# ---------------------------------------------------------------------------
# Parameter name heuristics (SSM / recurrent core)
# ---------------------------------------------------------------------------

# Prefer longer / more specific tokens first. Single-letter tokens are
# matched only as dotted path components (e.g. ".A.", "A_log").
_SSM_ALLOW_SUBSTRINGS = (
    "A_log", "dt_proj", "x_proj", "in_proj", "conv1d",
    "selective", "ssm", "delta", "decay", "state_size",
    "dt", "recurrent",
)
_SSM_SINGLE_LETTER = ("A", "B", "C", "D")
_SSM_BLOCK_SUBSTRINGS = (
    "bias", "norm", "gate", "lm_head", "embed", "attn",
)


def is_ssm_core_param(
    name: str,
    policies: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Heuristic: is this parameter part of an SSM / recurrent core?"""
    lower = name.lower()
    if policies:
        allow = policies.get("allow_substrings", _SSM_ALLOW_SUBSTRINGS)
        block = policies.get("block_substrings", _SSM_BLOCK_SUBSTRINGS)
        singles = policies.get("single_letter", _SSM_SINGLE_LETTER)
    else:
        allow = _SSM_ALLOW_SUBSTRINGS
        block = _SSM_BLOCK_SUBSTRINGS
        singles = _SSM_SINGLE_LETTER
    for b in block:
        if b.lower() in lower:
            return False
    for a in allow:
        if a.lower() in lower:
            return True
    # Single-letter tokens only as path components
    parts = lower.replace("/", ".").split(".")
    for s in singles:
        if s.lower() in parts:
            return True
    return False


def _validate_probability(name: str, value: float, inclusive_one: bool = True) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0 or (value > 1.0 if inclusive_one else value >= 1.0):
        raise ValueError(f"{name} must be in [0, 1{']' if inclusive_one else ')'}; got {value}")


def _validate_homogeneous_task_vectors(
    task_vectors: List[Dict[str, Tensor]],
) -> List[str]:
    if not task_vectors:
        raise ValueError("task_vectors must be non-empty")
    names = sorted(task_vectors[0].keys())
    if not names:
        raise ValueError("task vectors contain no parameters")
    for i, tv in enumerate(task_vectors):
        if sorted(tv.keys()) != names:
            raise ValueError(f"Expert {i} has mismatched parameter names")
        for n in names:
            if tv[n].shape != task_vectors[0][n].shape:
                raise ValueError(f"Shape mismatch for '{n}' in expert {i}")
    return names


def _normalize_expert_weights(
    weights: Tensor,
    num_experts: int,
    difficulty: Optional[Tensor] = None,
) -> Tensor:
    w = weights.float().reshape(-1)
    if w.numel() != num_experts:
        raise ValueError(f"weights length {w.numel()} != {num_experts}")
    if difficulty is not None:
        d = difficulty.float().reshape(-1)
        if d.numel() != num_experts:
            raise ValueError("difficulty length mismatch")
        w = w * (0.5 + 0.5 * d.clamp(0, 1))
    s = w.sum()
    if s <= 0 or not torch.isfinite(s):
        return torch.ones(num_experts, device=w.device) / num_experts
    return w / s


# ---------------------------------------------------------------------------
# Task-vector extraction
# ---------------------------------------------------------------------------

def extract_task_vectors(
    experts: Sequence[nn.Module],
    base: nn.Module,
) -> List[Dict[str, Tensor]]:
    """delta = expert − base for every shared floating-point parameter."""
    if not experts:
        raise ValueError("At least one expert is required")
    base_sd = {n: v.detach().clone() for n, v in base.state_dict().items()}
    task_vectors: List[Dict[str, Tensor]] = []

    for idx, expert in enumerate(experts):
        expert_sd = expert.state_dict()
        tv: Dict[str, Tensor] = {}
        for name, base_val in base_sd.items():
            exp_val = expert_sd.get(name)
            if exp_val is None or exp_val.shape != base_val.shape:
                continue
            if exp_val.is_floating_point() and base_val.is_floating_point():
                tv[name] = (exp_val.detach() - base_val).clone()
        if not tv:
            raise ValueError(f"Expert {idx} shares no floating-point parameters with base")
        task_vectors.append(tv)

    _validate_homogeneous_task_vectors(task_vectors)
    return task_vectors


# ---------------------------------------------------------------------------
# Stage 1 — DARE sparsification
# ---------------------------------------------------------------------------

def apply_dare_preprocessing(
    task_vectors: List[Dict[str, Tensor]],
    difficulty_importance: Optional[Tensor] = None,
    dare_base_p: float = 0.25,
    ssm_drop_reduction: float = 0.5,
    policies: Optional[Mapping[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[List[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
    """
    Randomly drop a fraction of delta entries and rescale the rest.

    Returns (processed_task_vectors, keep_masks).
    SSM-core parameters use a reduced drop rate.
    """
    _validate_probability("dare_base_p", dare_base_p, inclusive_one=False)
    _validate_probability("ssm_drop_reduction", ssm_drop_reduction)
    num_experts = len(task_vectors)
    _validate_homogeneous_task_vectors(task_vectors)

    if difficulty_importance is not None:
        difficulty = difficulty_importance.float().reshape(-1)
        if difficulty.numel() != num_experts:
            raise ValueError("difficulty_importance length mismatch")
    else:
        difficulty = None

    processed: List[Dict[str, Tensor]] = []
    keep_masks: List[Dict[str, Tensor]] = []

    for expert_index, task_vector in enumerate(task_vectors):
        p_effective = dare_base_p
        if difficulty is not None:
            importance = float(difficulty[expert_index].clamp(0.0, 1.0).item())
            p_effective = dare_base_p * (1.0 - 0.3 * importance)

        dropped: Dict[str, Tensor] = {}
        masks: Dict[str, Tensor] = {}
        for name, delta in task_vector.items():
            drop_rate = p_effective
            if is_ssm_core_param(name, policies):
                drop_rate *= ssm_drop_reduction
            drop_rate = min(0.95, max(0.0, float(drop_rate)))

            random_values = torch.rand(
                delta.shape,
                device=delta.device,
                dtype=torch.float32,
                generator=generator,
            )
            keep = random_values >= drop_rate
            scale = 1.0 / max(1.0 - drop_rate, 1e-8)
            dropped[name] = delta * keep.to(delta.dtype) * scale
            masks[name] = keep

        processed.append(dropped)
        keep_masks.append(masks)

    return processed, keep_masks


# ---------------------------------------------------------------------------
# Stage 2 — TIES v2 pure sign-majority
# ---------------------------------------------------------------------------

def difficulty_weighted_ties_merge(
    task_vectors: List[Dict[str, Tensor]],
    memory_bank_weights: Tensor,
    difficulty_importance: Optional[Tensor] = None,
    trim_ratio: float = 0.2,
    ssm_soft_merge: bool = True,
    policies: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Tensor]:
    """
    TIES merge with pure sign-majority election.

    Critical correction from v2.3:
      A heavy-magnitude outlier of one sign cannot overturn a majority of
      weaker experts of the opposite sign.  We elect the sign by *count*
      (optionally difficulty-weighted), then average only the agreeing experts.
    """
    _validate_probability("trim_ratio", trim_ratio, inclusive_one=False)
    names = _validate_homogeneous_task_vectors(task_vectors)
    num_experts = len(task_vectors)
    weights = _normalize_expert_weights(
        memory_bank_weights, num_experts, difficulty_importance
    )

    merged: Dict[str, Tensor] = {}
    for name in names:
        deltas = torch.stack([tv[name] for tv in task_vectors], dim=0)  # (E, ...)
        flat = deltas.flatten(1)  # (E, N)
        num_elements = flat.shape[1]

        # Magnitude trim (keep top-(1-trim) by |delta|)
        keep_count = max(1, math.ceil((1.0 - trim_ratio) * num_elements))
        kth = num_elements - keep_count + 1
        thresholds = flat.abs().kthvalue(kth, dim=1).values
        keep = flat.abs() >= thresholds.unsqueeze(1)
        trimmed = (flat * keep.to(flat.dtype)).view_as(deltas)

        # Soft merge for SSM cores (preserve continuous dynamics)
        if is_ssm_core_param(name, policies) and ssm_soft_merge:
            w = weights.view(num_experts, *([1] * (trimmed.dim() - 1)))
            merged[name] = (w * trimmed).sum(dim=0)
            continue

        # Pure sign-majority per element
        signs = torch.sign(trimmed)  # (E, ...)
        # Count of positive / negative (weight by expert importance)
        w = weights.view(num_experts, *([1] * (signs.dim() - 1)))
        pos_votes = (w * (signs > 0).float()).sum(dim=0)
        neg_votes = (w * (signs < 0).float()).sum(dim=0)
        elected_sign = torch.where(pos_votes >= neg_votes, 1.0, -1.0)

        # Average only experts that agree with the elected sign
        agree = (signs == elected_sign).float()
        agree_weight = (w * agree).sum(dim=0).clamp(min=1e-8)
        averaged = (w * agree * trimmed).sum(dim=0) / agree_weight
        # Zero positions where nothing survived trim
        averaged = torch.where(agree_weight > 1e-8, averaged, torch.zeros_like(averaged))
        merged[name] = averaged

    return merged


# ---------------------------------------------------------------------------
# Stage 3 — Fisher-weighted merge
# ---------------------------------------------------------------------------


def difficulty_weighted_ties_merge_with_masks(
    task_vectors: List[Dict[str, Tensor]],
    memory_bank_weights: Tensor,
    difficulty_importance: Optional[Tensor] = None,
    trim_ratio: float = 0.2,
    ssm_soft_merge: bool = True,
    policies: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
    """
    TIES sign-majority that also returns per-expert compatibility masks.

    masks[i][name] is True where expert i agrees with the elected sign
    (and survived magnitude trim). Soft-merged SSM params get all-True masks.
    """
    _validate_probability("trim_ratio", trim_ratio, inclusive_one=False)
    names = _validate_homogeneous_task_vectors(task_vectors)
    num_experts = len(task_vectors)
    weights = _normalize_expert_weights(
        memory_bank_weights, num_experts, difficulty_importance
    )

    merged: Dict[str, Tensor] = {}
    all_masks: List[Dict[str, Tensor]] = [{} for _ in range(num_experts)]

    for name in names:
        deltas = torch.stack([tv[name] for tv in task_vectors], dim=0)
        flat = deltas.flatten(1)
        num_elements = flat.shape[1]
        keep_count = max(1, math.ceil((1.0 - trim_ratio) * num_elements))
        kth = num_elements - keep_count + 1
        thresholds = flat.abs().kthvalue(kth, dim=1).values
        keep = flat.abs() >= thresholds.unsqueeze(1)
        trimmed = (flat * keep.to(flat.dtype)).view_as(deltas)

        if is_ssm_core_param(name, policies) and ssm_soft_merge:
            w = weights.view(num_experts, *([1] * (trimmed.dim() - 1)))
            merged[name] = (w * trimmed).sum(dim=0)
            for i in range(num_experts):
                all_masks[i][name] = torch.ones_like(trimmed[i], dtype=torch.bool)
            continue

        signs = torch.sign(trimmed)
        w = weights.view(num_experts, *([1] * (signs.dim() - 1)))
        pos_votes = (w * (signs > 0).float()).sum(dim=0)
        neg_votes = (w * (signs < 0).float()).sum(dim=0)
        elected_sign = torch.where(pos_votes >= neg_votes, 1.0, -1.0)

        agree = (signs == elected_sign) & (trimmed != 0)
        agree_f = agree.float()
        agree_weight = (w * agree_f).sum(dim=0).clamp(min=1e-8)
        averaged = (w * agree_f * trimmed).sum(dim=0) / agree_weight
        averaged = torch.where(agree_weight > 1e-8, averaged, torch.zeros_like(averaged))
        merged[name] = averaged
        for i in range(num_experts):
            all_masks[i][name] = agree[i]

    return merged, all_masks


def difficulty_weighted_fisher_merge(
    task_vectors: List[Dict[str, Tensor]],
    fisher_diagonals: List[Dict[str, Tensor]],
    memory_bank_weights: Tensor,
    difficulty_importance: Optional[Tensor] = None,
    dare_keep_masks: Optional[List[Dict[str, Tensor]]] = None,
    fisher_power: float = 1.0,
    fisher_floor: float = 1e-8,
) -> Dict[str, Tensor]:
    """
    Fisher-information weighted average of task vectors.

    Critical correction: when dare_keep_masks is supplied, the Fisher
    denominator *excludes* dropped elements so they do not dilute the
    importance of kept coordinates.
    """
    if not math.isfinite(fisher_power) or fisher_power <= 0:
        raise ValueError(f"fisher_power must be > 0; got {fisher_power}")
    if not math.isfinite(fisher_floor) or fisher_floor <= 0:
        raise ValueError(f"fisher_floor must be > 0; got {fisher_floor}")

    names = _validate_homogeneous_task_vectors(task_vectors)
    num_experts = len(task_vectors)
    if len(fisher_diagonals) != num_experts:
        raise ValueError("fisher_diagonals count mismatch")
    if dare_keep_masks is not None and len(dare_keep_masks) != num_experts:
        raise ValueError("dare_keep_masks count mismatch")

    weights = _normalize_expert_weights(
        memory_bank_weights, num_experts, difficulty_importance
    )

    merged: Dict[str, Tensor] = {}
    for name in names:
        numerator = torch.zeros_like(task_vectors[0][name], dtype=torch.float32)
        denominator = torch.zeros_like(task_vectors[0][name], dtype=torch.float32)

        for i, tv in enumerate(task_vectors):
            delta = tv[name].float()
            fish = fisher_diagonals[i].get(name)
            if fish is None:
                fish = torch.ones_like(delta)
            else:
                fish = fish.float().clamp(min=fisher_floor).pow(fisher_power)

            if dare_keep_masks is not None:
                keep = dare_keep_masks[i].get(name)
                if keep is not None:
                    fish = fish * keep.to(fish.dtype)

            w = float(weights[i])
            numerator = numerator + w * fish * delta
            denominator = denominator + w * fish

        merged[name] = numerator / denominator.clamp(min=fisher_floor)

    return merged


# ---------------------------------------------------------------------------
# Empirical Fisher diagonals
# ---------------------------------------------------------------------------

def _batch_size(batch: Any) -> int:
    if isinstance(batch, Tensor):
        return int(batch.shape[0])
    if isinstance(batch, Mapping):
        for v in batch.values():
            try:
                return _batch_size(v)
            except (TypeError, ValueError):
                continue
    if isinstance(batch, (tuple, list)):
        for v in batch:
            try:
                return _batch_size(v)
            except (TypeError, ValueError):
                continue
    raise ValueError("Unable to infer calibration batch size")


def _slice_batch(batch: Any, index: int) -> Any:
    if isinstance(batch, Tensor):
        return batch[index : index + 1]
    if isinstance(batch, Mapping):
        return {k: _slice_batch(v, index) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_slice_batch(v, index) for v in batch)
    if isinstance(batch, list):
        return [_slice_batch(v, index) for v in batch]
    return batch


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_batch_to_device(v, device) for v in batch)
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    return batch


def _default_forward(model: nn.Module, sample: Any) -> Any:
    if isinstance(sample, Mapping):
        return model(**sample)
    if isinstance(sample, (tuple, list)):
        return model(*sample)
    return model(sample)


def _extract_logits(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError("Unable to extract a Tensor from model output")


def _default_fisher_loss(output: Any, _: Any) -> Tensor:
    logits = _extract_logits(output)
    if logits.numel() == 0:
        raise ValueError("Model output is empty")
    if logits.dim() >= 2 and logits.shape[-1] >= 2:
        flat = logits.float().reshape(-1, logits.shape[-1])
        with torch.no_grad():
            targets = flat.detach().argmax(dim=-1)
        return F.cross_entropy(flat, targets)
    return logits.float().square().mean()


def build_empirical_fisher_diagonals(
    model: nn.Module,
    calibration_batch: StructuredBatch,
    forward_fn: Optional[ForwardFn] = None,
    loss_fn: Optional[LossFn] = None,
    device: Union[str, torch.device] = "cpu",
    micro_batch_size: int = 1,
) -> Dict[str, Tensor]:
    """
    Empirical diagonal Fisher via per-sample gradient accumulation.
    micro_batch_size=1 is rigorous (no cancellation inside the batch).
    """
    if _batch_size(calibration_batch) <= 0:
        raise ValueError("calibration_batch must contain at least one sample")
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")

    device = torch.device(device)
    original_training = model.training
    first_param = next(model.parameters(), None)
    original_device = first_param.device if first_param is not None else torch.device("cpu")

    fisher = {
        name: torch.zeros_like(p, device=device, dtype=torch.float32)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    batch = _move_batch_to_device(calibration_batch, device)
    batch_size = _batch_size(batch)
    forward = forward_fn or _default_forward
    loss_builder = loss_fn or _default_fisher_loss

    try:
        model.to(device)
        model.eval()
        n_accum = 0
        for start in range(0, batch_size, micro_batch_size):
            end = min(start + micro_batch_size, batch_size)
            for idx in range(start, end):
                sample = _slice_batch(batch, idx)
                model.zero_grad(set_to_none=True)
                output = forward(model, sample)
                loss = loss_builder(output, sample)
                loss.backward()
                for name, p in model.named_parameters():
                    if p.grad is not None and name in fisher:
                        fisher[name] += p.grad.detach().float().pow(2)
                n_accum += 1
        if n_accum > 0:
            for name in fisher:
                fisher[name] /= n_accum
    finally:
        model.zero_grad(set_to_none=True)
        model.to(original_device)
        model.train(original_training)

    return fisher


# ---------------------------------------------------------------------------
# High-level entry points
# ---------------------------------------------------------------------------

def merge_task_vectors_dare_ties_fisher(
    task_vectors: List[Dict[str, Tensor]],
    memory_bank_weights: Tensor,
    difficulty_importance: Optional[Tensor] = None,
    dare_base_p: float = 0.25,
    trim_ratio: float = 0.2,
    ssm_drop_reduction: float = 0.5,
    ssm_soft_merge: bool = True,
    fisher_diagonals: Optional[List[Dict[str, Tensor]]] = None,
    fisher_power: float = 1.0,
    policies: Optional[Mapping[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
    use_fisher: bool = True,
) -> Dict[str, Tensor]:
    """
    Correct multi-expert pipeline:

      1. DARE each expert independently → (Δ̃_i, M_dare_i)
      2. TIES sign-majority on DARE-processed vectors → compatibility via elect
         (when use_fisher=False, return TIES result directly)
      3. When Fisher is available: Fisher-weighted merge of DARE vectors with
         dare keep masks, so each expert's Fisher actually contributes.

    The previous implementation collapsed to a single TIES vector then applied
    only expert-0 Fisher — that is invalid and is no longer done.
    """
    processed, keep_masks = apply_dare_preprocessing(
        task_vectors,
        difficulty_importance=difficulty_importance,
        dare_base_p=dare_base_p,
        ssm_drop_reduction=ssm_drop_reduction,
        policies=policies,
        generator=generator,
    )
    # TIES always runs: produce elected result AND per-expert compatibility masks
    ties_result, ties_masks = difficulty_weighted_ties_merge_with_masks(
        processed,
        memory_bank_weights,
        difficulty_importance=difficulty_importance,
        trim_ratio=trim_ratio,
        ssm_soft_merge=ssm_soft_merge,
        policies=policies,
    )
    if not use_fisher or fisher_diagonals is None:
        return ties_result
    # Combine DARE keep masks with TIES sign-compatibility masks
    combined_masks: List[Dict[str, Tensor]] = []
    for i in range(len(processed)):
        cm: Dict[str, Tensor] = {}
        for name in processed[i]:
            d = keep_masks[i].get(name)
            t = ties_masks[i].get(name)
            if d is not None and t is not None:
                cm[name] = d.to(dtype=torch.bool) & t.to(dtype=torch.bool)
            elif d is not None:
                cm[name] = d.to(dtype=torch.bool)
            elif t is not None:
                cm[name] = t.to(dtype=torch.bool)
            else:
                cm[name] = torch.ones_like(processed[i][name], dtype=torch.bool)
        combined_masks.append(cm)
    return difficulty_weighted_fisher_merge(
        processed,
        fisher_diagonals,
        memory_bank_weights,
        difficulty_importance=difficulty_importance,
        dare_keep_masks=combined_masks,
        fisher_power=fisher_power,
    )


def apply_merged_task_vector(
    base: nn.Module,
    merged_delta: Dict[str, Tensor],
    alpha: float = 1.0,
) -> nn.Module:
    """Return a deep copy of `base` with merged_delta * alpha applied."""
    model = copy.deepcopy(base)
    sd = model.state_dict()
    for name, delta in merged_delta.items():
        if name in sd and sd[name].shape == delta.shape:
            sd[name] = sd[name] + alpha * delta.to(sd[name].device, sd[name].dtype)
    model.load_state_dict(sd)
    return model


def merge_expert_modules(
    experts: Sequence[nn.Module],
    base: nn.Module,
    memory_bank_weights: Optional[Tensor] = None,
    difficulty_importance: Optional[Tensor] = None,
    dare_base_p: float = 0.25,
    trim_ratio: float = 0.2,
    alpha: float = 1.0,
    policies: Optional[Mapping[str, Any]] = None,
    generator: Optional[torch.Generator] = None,
) -> nn.Module:
    """
    Convenience: extract task vectors → DARE → TIES → apply to a copy of base.
    """
    n = len(experts)
    if memory_bank_weights is None:
        memory_bank_weights = torch.ones(n)
    task_vectors = extract_task_vectors(experts, base)
    merged = merge_task_vectors_dare_ties_fisher(
        task_vectors,
        memory_bank_weights,
        difficulty_importance=difficulty_importance,
        dare_base_p=dare_base_p,
        trim_ratio=trim_ratio,
        policies=policies,
        generator=generator,
        use_fisher=False,
    )
    return apply_merged_task_vector(base, merged, alpha=alpha)
